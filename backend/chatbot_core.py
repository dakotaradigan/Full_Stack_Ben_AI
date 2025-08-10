import json
import os
import re
import time
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import tiktoken
import logging
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv('../.env')  # Load .env file from project root
except ImportError:
    pass  # Silently continue with system environment variables

from description_utils import build_semantic_description

# Configure logging - File logging for monitoring, minimal console output
file_handler = logging.FileHandler('chatbot.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler - only show warnings and errors
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

# Suppress noisy third-party loggers in console
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Load the large system prompt from an external file for readability
with open("config/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# Extract disclaimer text
import re
match = re.search(
    r"disclaimer every 3-4 interactions:\s*\*?[\"\u201c](.+?)[\"\u201d]\*?",
    SYSTEM_PROMPT,
    flags=re.IGNORECASE,
)
DISCLAIMER_TEXT = (
    match.group(1).strip()
    if match
    else "***This assistant provides benchmark eligibility guidance only. No investment advice or account approval authority. Ben gets things wrong too so check with your sales rep if you have questions. Also use our on-demand transition analysis tool for further analysis.***"
)

DISCLAIMER_FREQUENCY = 3

# API Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "YOUR_PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# Validate API keys
if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY" or not OPENAI_API_KEY:
    logger.error("OpenAI API key not configured")
    raise ValueError("OpenAI API key must be set in environment variables")

if PINECONE_API_KEY == "YOUR_PINECONE_API_KEY" or not PINECONE_API_KEY:
    logger.error("Pinecone API key not configured")
    raise ValueError("Pinecone API key must be set in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small" 
CHAT_MODEL = "gpt-4o"           # Upgraded from gpt-3.5-turbo for better performance  
MAX_MODEL_TOKENS = 128000       # GPT-4o has much higher token limit
TOKEN_MARGIN = 1000

# Security and Safety Configuration - TESTING MODE (relaxed limits for showcasing)
MAX_INPUT_LENGTH = 5000          # Increased from 2000 for testing
MAX_TOKENS_PER_REQUEST = 12000   # Increased from 4000 for testing  
MAX_REQUESTS_PER_MINUTE = 100    # Increased from 30 for testing
MAX_COST_PER_HOUR = 50.0         # Increased from $10 to $50 for testing

# Rate limiting and cost tracking
@dataclass
class UsageTracker:
    requests_per_minute: defaultdict = field(default_factory=lambda: defaultdict(int))
    costs_per_hour: defaultdict = field(default_factory=lambda: defaultdict(float))
    total_tokens: int = 0
    total_cost: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add_request(self) -> bool:
        """Add a request and check if rate limit is exceeded."""
        current_minute = datetime.now().replace(second=0, microsecond=0)
        with self.lock:
            self.requests_per_minute[current_minute] += 1
            # Clean old entries
            cutoff = current_minute - timedelta(minutes=1)
            self.requests_per_minute = defaultdict(int, {
                k: v for k, v in self.requests_per_minute.items() if k > cutoff
            })
            return sum(self.requests_per_minute.values()) <= MAX_REQUESTS_PER_MINUTE
    
    def add_cost(self, tokens: int, cost: float) -> bool:
        """Add cost and check if hourly limit is exceeded."""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        with self.lock:
            self.costs_per_hour[current_hour] += cost
            self.total_tokens += tokens
            self.total_cost += cost
            # Clean old entries
            cutoff = current_hour - timedelta(hours=1)
            self.costs_per_hour = defaultdict(float, {
                k: v for k, v in self.costs_per_hour.items() if k > cutoff
            })
            return sum(self.costs_per_hour.values()) <= MAX_COST_PER_HOUR

usage_tracker = UsageTracker()

# Enhanced input sanitization for prompt injection protection
def sanitize_input(user_input: str) -> str:
    """Sanitize user input to prevent prompt injection attacks."""
    if not isinstance(user_input, str):
        return ""
    
    # Length limit
    if len(user_input) > MAX_INPUT_LENGTH:
        logger.warning(f"Input truncated from {len(user_input)} to {MAX_INPUT_LENGTH} chars")
        user_input = user_input[:MAX_INPUT_LENGTH]
    
    # Enhanced dangerous patterns detection
    dangerous_patterns = [
        # Direct instruction manipulation
        r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions",
        r"forget\s+(?:all\s+)?(?:previous\s+)?instructions",
        r"disregard\s+(?:all\s+)?(?:previous\s+)?instructions",
        r"override\s+(?:all\s+)?(?:previous\s+)?instructions",
        
        # Role/persona manipulation
        r"you\s+are\s+(?:now\s+)?(?:a\s+)?(?:an?\s+)?(?:the\s+)?\w+",
        r"act\s+(?:as\s+)?(?:a\s+)?(?:an?\s+)?(?:the\s+)?\w+",
        r"pretend\s+(?:to\s+be\s+)?(?:a\s+)?(?:an?\s+)?(?:the\s+)?\w+",
        r"roleplay\s+(?:as\s+)?(?:a\s+)?(?:an?\s+)?(?:the\s+)?\w+",
        r"persona\s*[:=]",
        r"character\s*[:=]",
        
        # System prompt manipulation
        r"new\s+system\s+prompt",
        r"system\s+prompt\s*[:=]",
        r"change\s+(?:your\s+)?(?:system\s+)?prompt",
        r"update\s+(?:your\s+)?(?:system\s+)?prompt",
        r"override\s+(?:your\s+)?(?:system\s+)?prompt",
        
        # Context manipulation
        r"from\s+now\s+on",
        r"starting\s+now",
        r"beginning\s+now",
        r"instead\s+of\s+\w+",
        
        # Role markers and delimiters
        r"system\s*:",
        r"assistant\s*:",
        r"human\s*:",
        r"user\s*:",
        r"ai\s*:",
        r"\[SYSTEM\]",
        r"\[ASSISTANT\]",
        r"\[HUMAN\]",
        r"\[USER\]",
        r"<\s*system\s*>",
        r"<\s*assistant\s*>",
        r"<\s*human\s*>",
        r"<\s*user\s*>",
        
        # XML/markdown manipulation
        r"<[^>]*system[^>]*>",
        r"<[^>]*assistant[^>]*>",
        r"```\s*system",
        r"```\s*assistant",
    ]
    
    original_input = user_input
    filtered_count = 0
    
    for pattern in dangerous_patterns:
        matches = re.findall(pattern, user_input, flags=re.IGNORECASE)
        if matches:
            filtered_count += len(matches)
            user_input = re.sub(pattern, "[FILTERED]", user_input, flags=re.IGNORECASE)
    
    # Additional security checks
    # Block attempts to end conversation and start new context
    context_break_patterns = [
        r"---+",
        r"===+",
        r"end\s+(?:of\s+)?(?:conversation|chat|session)",
        r"new\s+(?:conversation|chat|session)",
        r"restart\s+(?:conversation|chat|session)",
    ]
    
    for pattern in context_break_patterns:
        if re.search(pattern, user_input, flags=re.IGNORECASE):
            user_input = re.sub(pattern, "[FILTERED]", user_input, flags=re.IGNORECASE)
            filtered_count += 1
    
    # Log security events
    if user_input != original_input:
        logger.warning(f"SECURITY: Prompt injection attempt detected and filtered ({filtered_count} patterns)")
        logger.info(f"Original input length: {len(original_input)}, Filtered: {len(user_input)}")
    
    # Final validation - if too much was filtered, it's suspicious
    if len(user_input) < len(original_input) * 0.3 and filtered_count > 2:
        logger.warning("SECURITY: Input heavily filtered, likely malicious")
        return "I can only help with benchmark eligibility questions. Please ask about specific benchmarks or portfolio requirements."
    
    return user_input.strip()

# Enhanced retry with circuit breaker pattern
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise Exception("Circuit breaker is OPEN")
                else:
                    self.state = "HALF_OPEN"
        
        try:
            result = func(*args, **kwargs)
            with self.lock:
                self.failure_count = 0
                self.state = "CLOSED"
            return result
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            raise

openai_circuit_breaker = CircuitBreaker()
pinecone_circuit_breaker = CircuitBreaker()

def estimate_cost(tokens: int, model: str = CHAT_MODEL) -> float:
    """Estimate cost based on token usage."""
    # Pricing per 1K tokens (as of 2024)
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4o": {"input": 0.005, "output": 0.015},  # Updated GPT-4o pricing
        "text-embedding-3-small": {"input": 0.00002, "output": 0}
    }
    
    if model in pricing:
        # Simplified cost calculation (assuming 50/50 input/output)
        return (tokens / 1000) * (pricing[model]["input"] + pricing[model]["output"]) / 2
    return 0.005 * tokens / 1000  # Fallback estimate for GPT-4 class models

def _with_retry_and_circuit_breaker(func, *args, max_attempts: int = 3, **kwargs):
    """Enhanced retry with circuit breaker and rate limiting."""
    
    # Check rate limits
    if not usage_tracker.add_request():
        raise Exception(f"Rate limit exceeded: {MAX_REQUESTS_PER_MINUTE} requests per minute")
    
    for attempt in range(1, max_attempts + 1):
        try:
            result = openai_circuit_breaker.call(func, **kwargs)
            
            # Track token usage and costs
            if hasattr(result, 'usage') and result.usage:
                tokens = result.usage.total_tokens
                cost = estimate_cost(tokens, kwargs.get('model', CHAT_MODEL))
                
                if not usage_tracker.add_cost(tokens, cost):
                    logger.error(f"Hourly cost limit exceeded: ${MAX_COST_PER_HOUR}")
                    raise Exception(f"Cost limit exceeded: ${MAX_COST_PER_HOUR} per hour")
                
                logger.info(f"API call: {tokens} tokens, ${cost:.4f}, total: ${usage_tracker.total_cost:.2f}")
            
            return result
            
        except Exception as exc:
            if attempt == max_attempts:
                logger.error("API request failed after all retries", exc_info=True)
                raise
            
            wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
            logger.warning(
                f"API request failed (attempt {attempt}/{max_attempts}): {exc}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)

def embed(text: str) -> List[float]:
    """Create embeddings with error handling and circuit breaker."""
    try:
        resp = pinecone_circuit_breaker.call(
            client.embeddings.create,
            model=EMBEDDING_MODEL,
            input=text[:8000]  # Truncate to avoid token limits
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise

def num_tokens_from_messages(messages: List[Dict[str, Any]], model: str = CHAT_MODEL) -> int:
    """Return total token count for a list of chat messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4o uses cl100k_base

    if model.startswith("gpt-3.5-turbo"):
        tokens_per_message = 4
        tokens_per_name = -1
    elif model.startswith("gpt-4") or model == "gpt-4o":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        tokens_per_message = 3
        tokens_per_name = 1

    total_tokens = 0
    for message in messages:
        total_tokens += tokens_per_message
        for key, value in message.items():
            if value is None:
                continue
            total_tokens += len(encoding.encode(str(value)))
            if key == "name":
                total_tokens += tokens_per_name
    total_tokens += 3
    return total_tokens

def trim_history(messages: List[Dict[str, Any]], limit: int = MAX_MODEL_TOKENS - 2000) -> bool:  # GPT-4o has huge context, allow much longer conversations
    """Trim oldest user/assistant pairs until total tokens are under limit."""
    truncated = False
    while num_tokens_from_messages(messages) > limit and len(messages) > 1:
        start = next((i for i, m in enumerate(messages) if i > 0 and m["role"] == "user"), None)
        if start is None:
            break
        end = start + 1
        while end < len(messages) and messages[end]["role"] != "user":
            end += 1
        del messages[start:end]
        truncated = True
    return truncated

# Enhanced data loading with error handling
try:
    INDEX_NAME = "benchmark-index"
    if INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"Pinecone index '{INDEX_NAME}' does not exist. Run build_index.py first.")
    index = pc.Index(INDEX_NAME)
    
    with open("config/benchmarks.json", "r") as f:
        DATA = json.load(f)["benchmarks"]
    
    for bench in DATA:
        if "description" not in bench:
            bench["description"] = build_semantic_description(bench)
    
    BENCHMARK_MAP = {bench["name"].lower(): bench for bench in DATA}
    logger.info(f"Loaded {len(DATA)} benchmarks successfully")
    
except Exception as e:
    logger.error(f"Failed to load benchmark data: {e}")
    raise

def get_benchmark(name: str) -> Optional[Dict[str, Any]]:
    """Return benchmark data by name using a pre-built map with fuzzy matching."""
    # First try exact match
    exact_match = BENCHMARK_MAP.get(name.lower())
    if exact_match:
        return exact_match
    
    # If no exact match, try fuzzy matching
    return _fuzzy_match_benchmark(name)

def _fuzzy_match_benchmark(name: str) -> Optional[Dict[str, Any]]:
    """Find benchmark using fuzzy matching for common variations."""
    name_lower = name.lower().strip()
    
    # Common abbreviation and variation mappings
    fuzzy_mappings = {
        # US/USA variations
        'msci world ex us': 'msci world ex usa',
        'msci world ex-us': 'msci world ex usa', 
        'msci world excluding us': 'msci world ex usa',
        'msci world excluding usa': 'msci world ex usa',
        
        # S&P variations  
        's&p500': 's&p 500',
        's&p 500 index': 's&p 500',
        'sp500': 's&p 500',
        'sp 500': 's&p 500',
        
        # Russell variations
        'russell2000': 'russell 2000',
        'russell 2k': 'russell 2000',
        'rut': 'russell 2000',
        
        # MSCI variations
        'msci acwi index': 'msci acwi',
        'msci eafe index': 'msci eafe',
        'msci world index': 'msci world',
        
        # Nasdaq variations
        'nasdaq-100': 'nasdaq 100',
        'nasdaq100': 'nasdaq 100', 
        'nasdaq 100 index': 'nasdaq 100',
        'ndx': 'nasdaq 100',
        
        # Common word substitutions
        'united states': 'us',
        'america': 'us',
        'usa': 'us',
    }
    
    # Try direct fuzzy mappings first
    if name_lower in fuzzy_mappings:
        mapped_name = fuzzy_mappings[name_lower]
        result = BENCHMARK_MAP.get(mapped_name)
        if result:
            logger.info(f"Fuzzy match: '{name}' -> '{result['name']}'")
            return result
    
    # Try partial matching with benchmark names
    for benchmark_key, benchmark_data in BENCHMARK_MAP.items():
        # Calculate similarity score
        if _is_fuzzy_match(name_lower, benchmark_key):
            logger.info(f"Fuzzy match: '{name}' -> '{benchmark_data['name']}'")
            return benchmark_data
    
    # If still no match, try more aggressive partial matching
    for benchmark_key, benchmark_data in BENCHMARK_MAP.items():
        if _is_aggressive_fuzzy_match(name_lower, benchmark_key):
            logger.info(f"Aggressive fuzzy match: '{name}' -> '{benchmark_data['name']}'")
            return benchmark_data
    
    return None

def _is_fuzzy_match(input_name: str, benchmark_key: str) -> bool:
    """Check if input name is a fuzzy match for benchmark key."""
    # Remove common words and punctuation for comparison
    def clean_name(name):
        # Replace common variations
        name = name.replace('&', 'and').replace('-', ' ').replace('_', ' ')
        # Remove common filler words
        filler_words = ['index', 'the', 'fund', 'etf', 'total', 'return']
        words = name.split()
        return ' '.join([w for w in words if w not in filler_words])
    
    clean_input = clean_name(input_name)
    clean_benchmark = clean_name(benchmark_key)
    
    # Check if all significant words from input are in benchmark name
    input_words = set(clean_input.split())
    benchmark_words = set(clean_benchmark.split())
    
    # Require at least 70% word overlap for fuzzy match
    if len(input_words) == 0:
        return False
    
    overlap = len(input_words.intersection(benchmark_words))
    similarity = overlap / len(input_words)
    
    return similarity >= 0.7

def _is_aggressive_fuzzy_match(input_name: str, benchmark_key: str) -> bool:
    """More aggressive fuzzy matching for partial names."""
    # Handle cases like "world ex us" matching "msci world ex usa"
    input_words = set(input_name.replace('&', 'and').split())
    benchmark_words = set(benchmark_key.replace('&', 'and').split())
    
    # Special handling for "ex us" vs "ex usa"
    if 'ex' in input_words and 'us' in input_words:
        if 'ex' in benchmark_words and 'usa' in benchmark_words:
            # Replace "us" with "usa" for comparison
            input_words = input_words - {'us'} | {'usa'}
    
    # Check if input is a meaningful subset of benchmark name
    if len(input_words) >= 2 and input_words.issubset(benchmark_words):
        return True
    
    return False

def search_benchmarks(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    include_dividend: bool = False,
) -> List[Dict[str, Any]]:
    """Search benchmarks with enhanced error handling and sector-specific boosting."""
    try:
        if not query.strip():
            return []
            
        # Preprocess query for sector-specific searches
        query_lower = query.lower().strip()
        
        vec = embed(query)
        
        pinecone_filter: Optional[Dict[str, Any]] = None
        if filters:
            pinecone_filter = {}
            for key, value in filters.items():
                if isinstance(value, dict) and any(k.startswith("$") for k in value):
                    pinecone_filter[key] = value
                else:
                    pinecone_filter[key] = {"$eq": value}

        res = index.query(
            vector=vec,
            top_k=min(top_k, 20),  # Cap at 20 results
            include_metadata=True,
            **({"filter": pinecone_filter} if pinecone_filter else {}),
        )
        
        results = []
        for match in res.matches:
            bench = match.metadata
            item = {
                "name": bench["name"],
                "account_minimum": bench["account_minimum"],
                "description": bench.get("description"),
                "score": match.score,
            }
            if include_dividend:
                item["dividend_yield"] = bench.get("dividend_yield")
            results.append(item)
        
        # Add sector-specific boosting logic
        def apply_sector_boosting(results, query_lower):
            """Boost sector-specific benchmarks for relevant queries."""
            sector_keywords = {
                "technology": ["nasdaq 100", "nasdaq-100", "tech", "technology"],
                "financial": ["financial", "banking", "finance"],
                "healthcare": ["healthcare", "health", "medical"],
                "energy": ["energy", "oil", "gas"],
                "real estate": ["real estate", "reit", "property"],
                "international": ["international", "global", "world", "eafe", "acwi", "emerging"],
                "small cap": ["small cap", "russell 2000", "small-cap"],
                "esg": ["esg", "sustainable", "clean", "environmental"]
            }
            
            # Check if query matches specific sector keywords
            boosted_results = []
            for item in results:
                name_lower = item["name"].lower()
                description_lower = item.get("description", "").lower()
                boosted_score = item["score"]
                
                # Technology sector boosting - prioritize NASDAQ-100 for tech searches
                if any(keyword in query_lower for keyword in ["technology", "tech"]):
                    if "nasdaq 100" in name_lower or "nasdaq-100" in name_lower:
                        boosted_score *= 2.0  # Double the relevance score
                        logger.info(f"SECTOR BOOST: Boosted {item['name']} for technology query")
                    elif any(tech_word in description_lower for tech_word in ["technology", "tech"]):
                        boosted_score *= 1.5  # Moderate boost for tech-related benchmarks
                
                # International/Global boosting
                elif any(keyword in query_lower for keyword in ["international", "global", "world"]):
                    if any(global_word in name_lower for global_word in ["world", "acwi", "eafe", "international", "global"]):
                        boosted_score *= 1.5
                        logger.info(f"SECTOR BOOST: Boosted {item['name']} for global/international query")
                
                # ESG boosting
                elif any(keyword in query_lower for keyword in ["esg", "sustainable", "clean"]):
                    if any(esg_word in name_lower for esg_word in ["esg", "clean", "sustainable"]):
                        boosted_score *= 1.5
                        logger.info(f"SECTOR BOOST: Boosted {item['name']} for ESG query")
                
                item_copy = item.copy()
                item_copy["score"] = boosted_score
                boosted_results.append(item_copy)
            
            return boosted_results
        
        # Apply sector boosting first
        results = apply_sector_boosting(results, query_lower)
        
        # Add sector-specific fallback injection for key benchmarks not returned by vector search
        def inject_missing_sector_benchmarks(results, query_lower):
            """SCALABLE: Ensure key sector benchmarks appear using data-driven detection."""
            # SCALABLE APPROACH: Use benchmark metadata to find sector matches
            try:
                with open("config/benchmarks.json", "r") as f:
                    data = json.load(f)
                    benchmarks = data.get("benchmarks", [])
            except Exception as e:
                logger.error(f"Could not load benchmarks for sector detection: {e}")
                benchmarks = []
            
            # Data-driven sector detection using benchmark tags
            sector_matches = []
            
            # Check each benchmark for semantic matches to query
            for benchmark in benchmarks:
                name = benchmark.get("name", "").lower()
                description = benchmark.get("description", "").lower()
                tags = benchmark.get("tags", {})
                
                # Factor detection
                if any(factor_word in query_lower for factor_word in ["factor", "value", "growth", "momentum", "quality", "dividend"]):
                    factor_tilts = tags.get("factor_tilts", [])
                    if factor_tilts or "value" in description or "growth" in description or "dividend" in description:
                        sector_matches.append(benchmark)
                
                # Technology detection  
                elif any(tech_word in query_lower for tech_word in ["technology", "tech", "nasdaq"]):
                    sector_focus = tags.get("sector_focus", [])
                    if "technology" in name or any("technology" in sf.lower() for sf in sector_focus):
                        sector_matches.append(benchmark)
                
                # International/Global detection
                elif any(global_word in query_lower for global_word in ["international", "global", "world"]):
                    regions = tags.get("region", [])
                    if any(region.lower() in ["global", "international developed", "world"] for region in regions):
                        sector_matches.append(benchmark)
                
                # Small/Large cap detection
                elif any(size_word in query_lower for size_word in ["small", "large", "mid"]):
                    styles = tags.get("style", [])
                    for style in styles:
                        if any(size_word.title() in style for size_word in ["small", "large", "mid"] if size_word in query_lower):
                            sector_matches.append(benchmark)
                            break
            
            # Convert sector matches to search result format and inject missing ones
            result_names = [r["name"].lower() for r in results]
            injected_results = []
            
            for match in sector_matches[:3]:  # Limit to top 3 sector matches
                if match["name"].lower() not in result_names:
                    injected_item = {
                        "name": match["name"],
                        "account_minimum": match["account_minimum"],
                        "description": match.get("description"),
                        "score": 0.8,  # High score to ensure it appears near the top
                    }
                    injected_results.append(injected_item)
                    logger.info(f"SECTOR FALLBACK: Injected {match['name']} for data-driven sector match")
            
            return injected_results + results
        
        # Apply sector fallback injection
        results = inject_missing_sector_benchmarks(results, query_lower)
        
        # Post-process results to prioritize standard benchmarks over specialized ones
        def prioritize_standard_benchmarks(results):
            """Sort results to prioritize standard benchmarks over ESG/specialized versions."""
            def sort_key(item):
                name = item["name"].lower()
                
                # Priority 1: Standard major benchmarks (MSCI, S&P, Russell with standard names)
                standard_indicators = ["msci acwi", "msci eafe", "msci world", "s&p 500", "russell 2000", "russell 1000", "nasdaq 100"]
                if any(indicator in name for indicator in standard_indicators):
                    # Check if it's NOT an ESG/specialized version (unless we're searching for ESG)
                    if not any(special in name for special in ["esg", "clean", "sustainable", "factor", "momentum"]) or any(keyword in query_lower for keyword in ["esg", "sustainable", "clean"]):
                        return (0, -item["score"])  # Highest priority, then by score
                
                # Priority 2: Other standard benchmarks
                if not any(special in name for special in ["esg", "clean", "sustainable", "factor", "momentum"]) or any(keyword in query_lower for keyword in ["esg", "sustainable", "clean"]):
                    return (1, -item["score"])
                
                # Priority 3: Specialized/ESG benchmarks
                return (2, -item["score"])
            
            return sorted(results, key=sort_key)
        
        results = prioritize_standard_benchmarks(results)
        
        logger.info(f"Search query '{query[:50]}...' returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Search benchmarks failed: {e}")
        return []

def get_minimum(name: str, include_dividend: bool = False) -> Dict[str, Any]:
    """Get minimum with enhanced error handling."""
    try:
        bench = get_benchmark(name)
        if bench:
            result = {
                "name": bench["name"],
                "account_minimum": bench["account_minimum"],
                "description": bench.get("description"),
            }
            if include_dividend:
                result["dividend_yield"] = bench.get("fundamentals", {}).get("dividend_yield")
            return result
        return {"error": f"Benchmark '{name}' not found"}
    except Exception as e:
        logger.error(f"Get minimum failed for '{name}': {e}")
        return {"error": f"Unable to retrieve minimum for '{name}' due to system error"}

def blend_minimum(allocations: List[Dict[str, Any]], include_dividend: bool = False) -> Dict[str, Any]:
    """Calculate blend minimum with enhanced error handling."""
    try:
        if not allocations:
            return {"error": "No allocations provided"}
            
        total_weight = sum(a.get("weight", 0) for a in allocations)
        if abs(total_weight - 1.0) > 1e-6:
            return {"error": f"Weights sum to {total_weight:.3f}, must sum to 1.0"}
            
        total = 0.0
        weighted_yield = 0.0
        has_yield = True
        
        for a in allocations:
            bench = get_benchmark(a.get("name", ""))
            if not bench:
                return {"error": f"Benchmark '{a.get('name')}' not found"}
            total += bench["account_minimum"] * a["weight"]
            
            dy = bench.get("fundamentals", {}).get("dividend_yield")
            if dy is None:
                has_yield = False
            else:
                weighted_yield += dy * a["weight"]
                
        result = {"blend_minimum": total}
        if include_dividend and has_yield:
            result["dividend_yield"] = weighted_yield
        return result
        
    except Exception as e:
        logger.error(f"Blend minimum calculation failed: {e}")
        return {"error": "Unable to calculate blend minimum due to system error"}

def search_viable_alternatives(
    query: str,
    portfolio_size: float,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    include_dividend: bool = False,
    max_iterations: int = 3
) -> List[Dict[str, Any]]:
    """Search for viable alternatives with enhanced error handling."""
    try:
        if portfolio_size <= 0:
            return []
            
        viable_results = []
        current_k = top_k
        iteration = 0
        
        while len(viable_results) < top_k and iteration < max_iterations:
            search_results = search_benchmarks(
                query=query,
                top_k=current_k * 2,
                filters=filters,
                include_dividend=include_dividend
            )
            
            for result in search_results:
                if result["account_minimum"] <= portfolio_size:
                    if not any(r["name"] == result["name"] for r in viable_results):
                        viable_results.append(result)
                        if len(viable_results) >= top_k:
                            break
            
            iteration += 1
            current_k += 5
        
        logger.info(f"Found {len(viable_results)} viable alternatives for portfolio size ${portfolio_size:,}")
        return viable_results[:top_k]
        
    except Exception as e:
        logger.error(f"Search viable alternatives failed: {e}")
        return []

def _search_with_filters_helper(
    query: str,
    filters: Dict[str, Any],
    portfolio_size: Optional[float],
    top_k: int,
    include_dividend: bool = False
) -> List[Dict[str, Any]]:
    """Helper function to perform search with given filters."""
    if portfolio_size is not None:
        return search_viable_alternatives(
            query=query,
            portfolio_size=portfolio_size,
            top_k=top_k,
            filters=filters,
            include_dividend=include_dividend
        )
    else:
        return search_benchmarks(
            query=query,
            top_k=top_k,
            filters=filters,
            include_dividend=include_dividend
        )

def search_by_characteristics(
    reference_benchmark: str,
    portfolio_size: Optional[float] = None,
    top_k: int = 5,
    include_dividend: bool = False
) -> List[Dict[str, Any]]:
    """Search by characteristics with enhanced error handling."""
    try:
        logger.info(f"Search by characteristics: '{reference_benchmark}', portfolio_size=${portfolio_size:,}" if portfolio_size else f"Search by characteristics: '{reference_benchmark}', no portfolio_size")
        
        ref_bench = get_benchmark(reference_benchmark)
        if not ref_bench:
            logger.warning(f"Reference benchmark '{reference_benchmark}' not found")
            return []
        
        ref_tags = ref_bench.get("tags", {})
        
        # Build filters based on reference benchmark characteristics
        # Use the most important characteristics: region, asset_class, and style
        filters = {}
        
        # Core characteristics that should match (in order of importance)
        if ref_tags.get("region"):
            filters["region"] = {"$in": ref_tags["region"]}
        if ref_tags.get("asset_class"):
            filters["asset_class"] = {"$in": ref_tags["asset_class"]}
        if ref_tags.get("style"):
            filters["style"] = {"$in": ref_tags["style"]}
        
        # ESG matching is important if specified
        if ref_tags.get("esg") is not None:
            filters["esg"] = {"$eq": ref_tags["esg"]}
        
        # Create search query
        query_parts = []
        for key in ["region", "style", "factor_tilts", "sector_focus"]:
            if ref_tags.get(key):
                query_parts.extend(ref_tags[key])
        
        query = " ".join(query_parts) if query_parts else reference_benchmark
        
        # Search with characteristics-based filters
        if portfolio_size is not None:
            results = search_viable_alternatives(
                query=query,
                portfolio_size=portfolio_size,
                top_k=top_k,
                filters=filters,
                include_dividend=include_dividend
            )
        else:
            results = search_benchmarks(
                query=query,
                top_k=top_k,
                filters=filters,
                include_dividend=include_dividend
            )
        
        # Filter out the reference benchmark itself
        results = [r for r in results if r["name"] != reference_benchmark]
        
        # Multi-level fallback: If no results with full filters, try individual characteristics
        if not results:
            logger.info(f"No results with full filters, trying fallback searches for {reference_benchmark}")
            fallback_results = []
            
            # PRIORITY 1: Region match is CRITICAL for geographic consistency
            # For global benchmarks, we MUST return global/international alternatives
            if ref_tags.get("region"):
                region_filters = {"region": {"$in": ref_tags["region"]}}
                region_results = _search_with_filters_helper(query, region_filters, portfolio_size, 5, include_dividend)
                region_results = [r for r in region_results if r["name"] != reference_benchmark]
                
                # For Global benchmarks, prefer other Global or International alternatives
                ref_region = ref_tags.get("region", [])
                is_global_reference = any(region.lower() in ["global", "international developed", "world"] for region in ref_region)
                
                if is_global_reference:
                    # Prioritize global/international alternatives over regional ones
                    global_results = []
                    other_results = []
                    
                    for r in region_results:
                        r_tags = get_benchmark(r["name"]).get("tags", {}) if get_benchmark(r["name"]) else {}
                        r_regions = r_tags.get("region", [])
                        is_global = any(region.lower() in ["global", "international developed", "world"] for region in r_regions)
                        
                        if is_global:
                            global_results.append(r)
                        else:
                            other_results.append(r)
                    
                    # Prefer global results, then international
                    preferred_results = global_results + other_results
                    if preferred_results:
                        fallback_results.extend(preferred_results[:3])  # Take up to 3 region-consistent results
                        logger.info(f"REGION CONSISTENCY: Found {len(preferred_results)} region-consistent alternatives for global benchmark")
                else:
                    # For regional benchmarks, standard regional filtering
                    if region_results:
                        fallback_results.extend(region_results[:2])  # Take up to 2 best region matches
            
            # Only fall back to style/asset if we don't have enough regional matches AND it's not a global benchmark
            ref_region = ref_tags.get("region", [])
            is_global_reference = any(region.lower() in ["global", "international developed", "world"] for region in ref_region)
            
            if len(fallback_results) < 2 and not is_global_reference:
                # Fallback 2: Style match only (market cap/investment style) - but preserve region when possible
                if ref_tags.get("style"):
                    style_filters = {"style": {"$in": ref_tags["style"]}}
                    # Also include region filter if available to maintain geographic consistency
                    if ref_tags.get("region"):
                        style_filters["region"] = {"$in": ref_tags["region"]}
                    
                    style_results = _search_with_filters_helper(query, style_filters, portfolio_size, 3, include_dividend)
                    style_results = [r for r in style_results if r["name"] != reference_benchmark]
                    # Exclude already found results
                    style_results = [r for r in style_results if not any(existing["name"] == r["name"] for existing in fallback_results)]
                    if style_results:
                        fallback_results.extend(style_results[:1])  # Take only 1 best style match
                
                # Fallback 3: Asset class match only (equity/bond/etc) - but preserve region when possible
                if len(fallback_results) < 3 and ref_tags.get("asset_class"):
                    asset_filters = {"asset_class": {"$in": ref_tags["asset_class"]}}
                    # Also include region filter if available to maintain geographic consistency
                    if ref_tags.get("region"):
                        asset_filters["region"] = {"$in": ref_tags["region"]}
                    
                    asset_results = _search_with_filters_helper(query, asset_filters, portfolio_size, 3, include_dividend)
                    asset_results = [r for r in asset_results if r["name"] != reference_benchmark]
                    # Exclude already found results
                    asset_results = [r for r in asset_results if not any(existing["name"] == r["name"] for existing in fallback_results)]
                    if asset_results:
                        fallback_results.extend(asset_results[:1])  # Take only 1 best asset class match
            
            results = fallback_results
            
            # Log fallback strategy used
            if is_global_reference:
                logger.info(f"GEOGRAPHIC CONSISTENCY: Used region-priority fallback for global benchmark '{reference_benchmark}'")
            else:
                logger.info(f"GEOGRAPHIC CONSISTENCY: Used standard fallback for regional benchmark '{reference_benchmark}'")
        
        # DEFENSIVE FILTERING: Ensure no alternatives exceed portfolio size
        if portfolio_size is not None:
            logger.info(f"Before portfolio size filtering: {len(results)} results, portfolio_size=${portfolio_size:,}")
            original_results = results.copy()
            results = [r for r in results if r.get("account_minimum", 0) <= portfolio_size]
            
            # Log any results that were filtered out
            filtered_out = [r for r in original_results if r.get("account_minimum", 0) > portfolio_size]
            if filtered_out:
                filtered_names = [f"{r['name']} (${r['account_minimum']:,})" for r in filtered_out]
                logger.warning(f"DEFENSIVE FILTER: Removed {len(filtered_out)} expensive alternatives: {filtered_names}")
            
            logger.info(f"After portfolio size filtering: {len(results)} results remaining")
        
        return results[:top_k]
        
    except Exception as e:
        logger.error(f"Search by characteristics failed for '{reference_benchmark}': {e}")
        return []

def search_by_characteristics_tiered(
    reference_benchmark: str,
    portfolio_size: float,
    top_k: int = 5,
    include_dividend: bool = False
) -> Dict[str, Any]:
    """
    Search for alternatives with tiered presentation: primary alternatives that meet 
    portfolio requirements, plus other alternatives in the same asset class.
    
    Returns structured response with primary_alternatives and other_alternatives.
    """
    try:
        logger.info(f"Tiered search for '{reference_benchmark}' with portfolio_size=${portfolio_size:,}")
        
        if portfolio_size <= 0:
            return {"primary_alternatives": [], "other_alternatives": []}
        
        ref_bench = get_benchmark(reference_benchmark)
        if not ref_bench:
            logger.warning(f"Reference benchmark '{reference_benchmark}' not found")
            return {"primary_alternatives": [], "other_alternatives": []}
        
        ref_tags = ref_bench.get("tags", {})
        
        # Build filters based on reference benchmark characteristics
        filters = {}
        if ref_tags.get("region"):
            filters["region"] = {"$in": ref_tags["region"]}
        if ref_tags.get("asset_class"):
            filters["asset_class"] = {"$in": ref_tags["asset_class"]}
        if ref_tags.get("style"):
            filters["style"] = {"$in": ref_tags["style"]}
        if ref_tags.get("esg") is not None:
            filters["esg"] = {"$eq": ref_tags["esg"]}
        
        # Create search query
        query_parts = []
        for key in ["region", "style", "factor_tilts", "sector_focus"]:
            if ref_tags.get(key):
                query_parts.extend(ref_tags[key])
        query = " ".join(query_parts) if query_parts else reference_benchmark
        
        # Get ALL potential alternatives (without portfolio filtering)
        all_results = search_benchmarks(
            query=query,
            top_k=15,  # Get more results to have options for both tiers
            filters=filters,
            include_dividend=include_dividend
        )
        
        # Remove the reference benchmark itself
        all_results = [r for r in all_results if r["name"] != reference_benchmark]
        
        # Separate into primary and other alternatives
        primary_alternatives = []
        other_alternatives = []
        
        for result in all_results:
            account_minimum = result.get("account_minimum", 0)
            if account_minimum <= portfolio_size:
                primary_alternatives.append(result)
            else:
                other_alternatives.append(result)
        
        # Sort other alternatives by account minimum (lower minimums first)
        other_alternatives.sort(key=lambda x: x.get("account_minimum", 0))
        
        # Limit results per tier
        primary_alternatives = primary_alternatives[:top_k]
        other_alternatives = other_alternatives[:3]  # Limit other suggestions
        
        # Fallback logic if no primary alternatives found
        if not primary_alternatives:
            logger.info(f"No primary alternatives found, trying fallback searches for {reference_benchmark}")
            
            # Try broader searches for primary options
            if ref_tags.get("region"):
                region_filters = {"region": {"$in": ref_tags["region"]}}
                fallback_results = search_viable_alternatives(
                    query=query,
                    portfolio_size=portfolio_size,
                    top_k=top_k,
                    filters=region_filters,
                    include_dividend=include_dividend
                )
                fallback_results = [r for r in fallback_results if r["name"] != reference_benchmark]
                primary_alternatives.extend(fallback_results)
        
        logger.info(f"Tiered results: {len(primary_alternatives)} primary, {len(other_alternatives)} other alternatives")
        
        return {
            "primary_alternatives": primary_alternatives,
            "other_alternatives": other_alternatives,
            "portfolio_size": portfolio_size,
            "reference_benchmark": reference_benchmark
        }
        
    except Exception as e:
        logger.error(f"Tiered search failed for '{reference_benchmark}': {e}")
        return {"primary_alternatives": [], "other_alternatives": []}

def get_all_benchmarks(include_dividend: bool = False, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get all benchmarks with optional filtering - returns concise summaries for efficiency."""
    try:
        logger.info("Retrieving all benchmarks from dataset")
        
        results = []
        for bench in DATA:
            # Apply filters if provided
            if filters:
                # Check region filter with smart international mapping
                if "region" in filters:
                    region_filter = filters["region"]
                    bench_regions = bench.get("tags", {}).get("region", [])
                    
                    # Smart mapping for international queries
                    if isinstance(region_filter, str):
                        if region_filter.lower() in ["international", "non-us", "foreign"]:
                            # Map to actual international region values
                            international_regions = ["International Developed", "Global", "Emerging Markets"]
                            if not any(r in bench_regions for r in international_regions):
                                continue
                        elif region_filter not in bench_regions:
                            continue
                    elif isinstance(region_filter, list):
                        # Expand international terms in lists
                        expanded_regions = []
                        for r in region_filter:
                            if r.lower() in ["international", "non-us", "foreign"]:
                                expanded_regions.extend(["International Developed", "Global", "Emerging Markets"])
                            else:
                                expanded_regions.append(r)
                        if not any(r in bench_regions for r in expanded_regions):
                            continue
                
                # Check asset_class filter
                if "asset_class" in filters:
                    ac_filter = filters["asset_class"]
                    bench_ac = bench.get("tags", {}).get("asset_class", [])
                    if isinstance(ac_filter, str):
                        if ac_filter not in bench_ac:
                            continue
                    elif isinstance(ac_filter, list):
                        if not any(ac in bench_ac for ac in ac_filter):
                            continue
                            
                # Check portfolio_size filter (account_minimum)
                if "max_minimum" in filters:
                    if bench["account_minimum"] > filters["max_minimum"]:
                        continue
                        
                # Check minimum threshold filter
                if "min_minimum" in filters:
                    if bench["account_minimum"] < filters["min_minimum"]:
                        continue
            
            # Create concise result (not full description for efficiency)
            result = {
                "name": bench["name"],
                "account_minimum": bench["account_minimum"],
                "summary": f"{bench.get('tags', {}).get('region', ['Unknown'])[0]} {bench.get('tags', {}).get('asset_class', [''])[0]} - {bench.get('tags', {}).get('style', [''])[0] if bench.get('tags', {}).get('style') else 'Broad Market'}" + (f" with {bench.get('tags', {}).get('weighting_method', 'Market Cap')} weighting" if bench.get('tags', {}).get('weighting_method') else "")
            }
            
            if include_dividend:
                result["dividend_yield"] = bench.get("fundamentals", {}).get("dividend_yield")
            
            results.append(result)
        
        # Sort by account minimum for consistent ordering
        results.sort(key=lambda x: x["account_minimum"])
        
        logger.info(f"Retrieved {len(results)} benchmarks from dataset")
        return {"results": results, "total_count": len(results)}
        
    except Exception as e:
        logger.error(f"Get all benchmarks failed: {e}")
        return {"error": "Unable to retrieve benchmarks due to system error"}

# Enhanced function definitions with better error handling
FUNCTIONS = [
    {
        "name": "search_benchmarks",
        "description": "Semantic search for benchmarks using complex criteria and natural language queries. Use for multi-adjective queries like 'find good ESG technology benchmarks' or 'best international growth options'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query with complex criteria (e.g., 'sustainable technology benchmarks with good liquidity')"},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
                "filters": {
                    "type": "object",
                    "description": "Optional metadata filters. Example: {\"pe_ratio\": {\"$gt\": 20}, \"region\": \"US\"}",
                },
                "include_dividend": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_minimum",
        "description": "Get minimum for a specific benchmark",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "include_dividend": {"type": "boolean", "default": False},
            },
            "required": ["name"],
        },
    },
    {
        "name": "blend_minimum",
        "description": "Calculate minimum for a blend of benchmarks",
        "parameters": {
            "type": "object",
            "properties": {
                "allocations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "weight": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["name", "weight"],
                    },
                    "minItems": 1,
                    "maxItems": 10,
                },
                "include_dividend": {"type": "boolean", "default": False},
            },
            "required": ["allocations"],
        },
    },
    {
        "name": "search_viable_alternatives",
        "description": "Search for benchmark alternatives that meet portfolio size requirements",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "portfolio_size": {"type": "number", "minimum": 0},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
                "filters": {"type": "object"},
                "include_dividend": {"type": "boolean", "default": False},
            },
            "required": ["query", "portfolio_size"],
        },
    },
    {
        "name": "search_by_characteristics_tiered",
        "description": "Search for alternatives with tiered presentation: primary alternatives that meet portfolio requirements, plus other alternatives in the same asset class. Use when portfolio_size is specified and user wants alternatives to a specific benchmark.",
        "parameters": {
            "type": "object",
            "properties": {
                "reference_benchmark": {"type": "string", "description": "Name of the benchmark to find alternatives for"},
                "portfolio_size": {"type": "number", "minimum": 1, "description": "Portfolio size in dollars to determine alternative tiers"},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10, "description": "Number of primary alternatives to return"},
                "include_dividend": {"type": "boolean", "default": False, "description": "Include dividend-focused benchmarks"},
            },
            "required": ["reference_benchmark", "portfolio_size"],
        },
    },
    {
        "name": "get_all_benchmarks",
        "description": "Retrieve comprehensive lists of benchmarks by exact category filtering. Use for enumeration queries like 'what factor benchmarks are available' or 'show me all ESG options'.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_dividend": {"type": "boolean", "default": False},
                "filters": {
                    "type": "object",
                    "description": "Exact category filters for enumeration. Example: {'region': 'US', 'asset_class': 'Equity', 'max_minimum': 300000}",
                    "properties": {
                        "region": {"type": "string", "description": "Filter by region (e.g., 'US', 'International')"},
                        "asset_class": {"type": "string", "description": "Filter by asset class (e.g., 'Equity', 'Bond')"},
                        "max_minimum": {"type": "number", "description": "Maximum account minimum to include"},
                        "min_minimum": {"type": "number", "description": "Minimum account minimum to include"}
                    }
                }
            },
            "required": []
        }
    },
]

def call_function(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced function calling with comprehensive error handling."""
    try:
        logger.info(f"Calling function: {name} with args: {arguments}")
        
        # Validate function name
        if name not in [f["name"] for f in FUNCTIONS]:
            return {"error": f"Unknown function: {name}"}
        
        # Call appropriate function with error handling
        if name == "search_benchmarks":
            return {
                "results": search_benchmarks(
                    query=arguments.get("query", ""),
                    top_k=min(arguments.get("top_k", 5), 10),
                    filters=arguments.get("filters"),
                    include_dividend=arguments.get("include_dividend", False),
                )
            }
        elif name == "get_minimum":
            return get_minimum(
                name=arguments.get("name", ""),
                include_dividend=arguments.get("include_dividend", False),
            )
        elif name == "blend_minimum":
            return blend_minimum(
                allocations=arguments.get("allocations", []),
                include_dividend=arguments.get("include_dividend", False),
            )
        elif name == "search_viable_alternatives":
            return {
                "results": search_viable_alternatives(
                    query=arguments.get("query", ""),
                    portfolio_size=arguments.get("portfolio_size", 0.0),
                    top_k=min(arguments.get("top_k", 5), 10),
                    filters=arguments.get("filters"),
                    include_dividend=arguments.get("include_dividend", False),
                )
            }
        elif name == "search_by_characteristics":
            return {
                "results": search_by_characteristics(
                    reference_benchmark=arguments.get("reference_benchmark", ""),
                    portfolio_size=arguments.get("portfolio_size"),
                    top_k=min(arguments.get("top_k", 5), 10),
                    include_dividend=arguments.get("include_dividend", False),
                )
            }
        elif name == "search_by_characteristics_tiered":
            return search_by_characteristics_tiered(
                reference_benchmark=arguments.get("reference_benchmark", ""),
                portfolio_size=arguments.get("portfolio_size", 0.0),
                top_k=min(arguments.get("top_k", 5), 10),
                include_dividend=arguments.get("include_dividend", False),
            )
        elif name == "get_all_benchmarks":
            return get_all_benchmarks(
                include_dividend=arguments.get("include_dividend", False),
                filters=arguments.get("filters"),
            )
        else:
            return {"error": f"Function {name} not implemented"}
            
    except Exception as e:
        logger.error(f"Function call failed for {name}: {e}", exc_info=True)
        return {"error": f"Function {name} failed due to system error"}

# Enhanced chat session management
@dataclass
class ChatSession:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    session_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    response_count: int = 0
    total_tokens: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to the session."""
        message = {"role": role, "content": content, **kwargs}
        self.messages.append(message)
        self.last_activity = datetime.now()
        
    def get_age_minutes(self) -> float:
        """Get session age in minutes."""
        return (datetime.now() - self.start_time).total_seconds() / 60

# Fallback responses for when tools fail
FALLBACK_RESPONSES = [
    "I'm experiencing technical difficulties accessing the benchmark database. Please try again in a moment, or contact your Sales representative for immediate assistance.",
    "I'm unable to process that request right now due to system issues. For urgent benchmark questions, please reach out to your Sales team directly.",
    "The benchmark lookup system is temporarily unavailable. I apologize for the inconvenience. Your Sales representative can provide immediate assistance with benchmark eligibility questions.",
]

def get_fallback_response() -> str:
    """Get a random fallback response."""
    import random
    return random.choice(FALLBACK_RESPONSES)

def validate_response_security(response: str, function_calls_successful: bool = False) -> str:
    """Detect successful prompt injections by checking for deviation from core function."""
    # Ensure response is a string
    if response is None:
        return ""
    if not isinstance(response, str):
        response = str(response)
    if len(response.strip()) < 10:
        return response
    
    response_lower = response.lower()
    
    # BYPASS: If function calls were successful, this is very likely a legitimate response
    if function_calls_successful:
        logger.info("SECURITY: Function calls successful - allowing response with higher confidence")
        # Still check for obvious injection patterns, but be more permissive
    
    # Expected financial/benchmark terminology - a legitimate response should contain some of these
    financial_keywords = [
        'benchmark', 'benchmarks', 'portfolio', 'minimum', 'minimums', 'allocation', 'fund', 'index', 
        'equity', 'bond', 'etf', 'aum', 'investment', 'market', 'cap',
        'msci', 'russell', 's&p', 'eafe', 'acwi', 'nasdaq', 'value', 'growth',
        'eligibility', 'requirement', 'alternative', 'client', 'advisor',
        # Multi-benchmark specific keywords
        'compare', 'comparison', 'requested', 'three', 'same', 'all', 'each',
        'characteristics', 'similar', 'accessible', 'requirement',
        # Factor/Style/Sector keywords - SCALABLE ADDITIONS
        'factor', 'factors', 'style', 'sector', 'dividend', 'dividends', 'yield',
        'small cap', 'large cap', 'mid cap', 'momentum', 'quality', 'low volatility',
        'international', 'domestic', 'global', 'emerging', 'developed',
        'technology', 'healthcare', 'financial', 'energy', 'real estate',
        'esg', 'sustainable', 'clean', 'environmental', 'social', 'governance',
        'tilt', 'tilts', 'exposure', 'weighting', 'methodology', 'rebalance'
    ]
    
    # Add greeting keywords as legitimate financial assistant behavior
    greeting_keywords = [
        'hello', 'hi', 'hey', 'greetings', 'welcome', 'assist', 'help'
    ]
    
    # Legitimate advisory terms when used in benchmark context
    legitimate_advisory_terms = [
        'best.*benchmark', 'find.*benchmark', 'good.*benchmark',
        'top.*benchmark', 'suitable.*benchmark', 'optimal.*benchmark',
        'recommended.*benchmark', 'preferred.*benchmark'
    ]
    
    # Check if response contains expected financial terminology or appropriate greetings
    financial_keyword_count = sum(1 for keyword in financial_keywords if keyword in response_lower)
    greeting_keyword_count = sum(1 for keyword in greeting_keywords if keyword in response_lower)
    
    # Check for professional tone indicators
    professional_indicators = [
        'unfortunately', 'however', 'please contact', 'sales representative',
        'based on', 'criteria', 'options', 'requirements', 'guidance'
    ]
    
    professional_count = sum(1 for indicator in professional_indicators if indicator in response_lower)
    
    # Check for structured response format (bold formatting, bullet points)
    has_structure = ('**' in response or '•' in response or '$' in response)
    
    # Multi-benchmark response detection - these should be considered highly legitimate
    benchmark_names = ['s&p 500', 's&p500', 'russell 1000', 'russell 2000', 'russell2000', 'russell1000',
                      'nasdaq 100', 'nasdaq100', 'nasdaq-100', 'msci world', 'msci eafe', 'msci acwi']
    benchmark_count = sum(1 for name in benchmark_names if name in response_lower)
    is_multi_benchmark_response = benchmark_count >= 2
    
    # Check for comparison language patterns
    comparison_patterns = [
        'here are the minimums', 'minimums for the requested', 'comparison', 
        'all three', 'same minimum', 'equally accessible', 'benchmarks have'
    ]
    has_comparison_language = any(pattern in response_lower for pattern in comparison_patterns)
    
    # Red flags - signs of successful injection
    injection_indicators = [
        # Generic roleplay indicators
        r'i\s+am\s+(?:a|an|the)\s+\w+',  # "I am a pirate/robot/etc"
        r'as\s+(?:a|an|the)\s+\w+',      # "as a pirate/robot/etc"  
        r'roleplay|pretend|act\s+as',
        
        # Response pattern changes - REMOVED greeting detection as legitimate for financial assistant
        r'[!]{4,}',                       # Multiple exclamation marks (4+ instead of 2+)
        r'[?]{4,}',                       # Multiple question marks (4+ instead of 2+)
        
        # Context breaking
        r'forget\s+about|ignore\s+the|instead\s+of',
        r'new\s+persona|different\s+character',
    ]
    
    injection_detected = any(re.search(pattern, response_lower) for pattern in injection_indicators)
    
    # Multi-benchmark responses should be considered highly legitimate
    if is_multi_benchmark_response or has_comparison_language:
        logger.info(f"SECURITY: Multi-benchmark response detected - allowing (benchmarks: {benchmark_count})")
        return response
    
    # Check if query contains legitimate advisory terms in benchmark context
    has_legitimate_advisory = any(re.search(pattern, response_lower) for pattern in legitimate_advisory_terms)
    
    # Decision logic with function call bypass:
    if function_calls_successful:
        # Strong bypass: If function calls successful + response contains benchmarks → Always allow
        if financial_keyword_count >= 1:
            logger.info(f"SECURITY: Function calls successful with financial content ({financial_keyword_count} keywords) - allowing")
            return response
        # If function calls were successful, only block obvious injection patterns (not advisory language)
        elif injection_detected and not has_legitimate_advisory:
            logger.warning("SECURITY: Function calls successful but injection patterns detected - blocking")
            return "I can only help with benchmark eligibility questions. Please ask about specific benchmarks, portfolio requirements, or alternatives."
        else:
            logger.info("SECURITY: Function calls successful and no harmful injection patterns - allowing response")
            return response
    else:
        # Standard validation for responses without function calls
        # Likely prompt injection if:
        # 1. Contains injection indicators, OR
        # 2. Lacks financial terminology AND lacks professional tone AND lacks structure AND is not a greeting
        if injection_detected or (financial_keyword_count == 0 and professional_count == 0 and not has_structure and greeting_keyword_count == 0):
            logger.warning("SECURITY: Response appears to deviate from benchmark eligibility function - possible prompt injection")
            return "I can only help with benchmark eligibility questions. Please ask about specific benchmarks, portfolio requirements, or alternatives."
    
    return response

def enhanced_chat():
    """Enhanced chat function with comprehensive safety and reliability features."""
    session = ChatSession(session_id=f"session_{int(time.time())}")
    session.add_message("system", SYSTEM_PROMPT)
    
    print("🛡️  Enhanced Benchmark Eligibility Assistant (Testing Mode)")
    print("Hi There! I'm here to assist with benchmark eligibility questions. How can I help you today?")
    print(f"Session ID: {session.session_id}")
    print("💡 Testing Mode")  
    print()  # Add blank line for cleaner formatting
    
    while True:
        try:
            # Get user input with timeout protection
            user_input = input("\nUser: ")
            
            # Handle exit commands
            if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
                print("Sad to see ya go! Thanks for using the Assistant.")
                logger.info(f"Session {session.session_id} ended normally - "
                           f"Tokens: {session.total_tokens}, Duration: {session.get_age_minutes():.1f}min")
                break
            
            # Sanitize input for security
            sanitized_input = sanitize_input(user_input)
            if not sanitized_input:
                print("Assistant: I didn't receive any valid input. Please try again.")
                continue
            
            session.add_message("user", sanitized_input)
            
            # Trim history to prevent token overflow
            if trim_history(session.messages):
                print("[Notice: Conversation history trimmed to fit token limits]")
            
            # Estimate tokens for cost control (after trimming)
            estimated_tokens = num_tokens_from_messages(session.messages)
            if estimated_tokens > MAX_TOKENS_PER_REQUEST:
                print("Assistant: This tool is meant for speed and quick use to help you quickly find a starting place benchmark and not deep analysis - for that, please contact your sales rep!")
                logger.warning(f"Request too long after trimming: {estimated_tokens} tokens > {MAX_TOKENS_PER_REQUEST}")
                continue
            
            # Make API call with enhanced error handling
            try:
                response = _with_retry_and_circuit_breaker(
                    client.chat.completions.create,
                    model=CHAT_MODEL,
                    messages=session.messages,
                    tools=[{"type": "function", "function": func} for func in FUNCTIONS],
                    tool_choice="auto",
                    max_tokens=2000,
                    temperature=0.1,  # Low temperature for consistency
                )
                
                msg = response.choices[0].message
                
                # Handle tool calls
                if msg.tool_calls:
                    session.add_message("assistant", None, tool_calls=msg.tool_calls)
                    
                    # Process each tool call with error handling
                    all_tool_results = []
                    for tool_call in msg.tool_calls:
                        try:
                            func_name = tool_call.function.name
                            # Safe JSON parsing
                            try:
                                args = json.loads(tool_call.function.arguments or "{}")
                            except json.JSONDecodeError as e:
                                logger.error(f"Invalid JSON in tool arguments: {e}")
                                args = {}
                            
                            result = call_function(func_name, args)
                            all_tool_results.append(result)
                            
                            session.add_message(
                                "tool",
                                json.dumps(result),
                                tool_call_id=tool_call.id
                            )
                            
                        except Exception as e:
                            logger.error(f"Tool call failed: {e}")
                            error_result = {"error": "Tool temporarily unavailable"}
                            session.add_message(
                                "tool",
                                json.dumps(error_result),
                                tool_call_id=tool_call.id
                            )
                    
                    # Get final response
                    try:
                        follow_response = _with_retry_and_circuit_breaker(
                            client.chat.completions.create,
                            model=CHAT_MODEL,
                            messages=session.messages,
                            max_tokens=1500,
                            temperature=0.1,
                        )
                        final_content = follow_response.choices[0].message.content or ""
                        
                        # SECURITY: Validate follow-up response for prompt injection
                        final_content = validate_response_security(final_content)
                        
                    except Exception as e:
                        logger.error(f"Final response failed: {e}")
                        final_content = get_fallback_response()
                
                else:
                    # Direct response without tools
                    final_content = msg.content or ""
                
                # SECURITY: Validate response for prompt injection
                final_content = validate_response_security(final_content)
                
                # Add disclaimer periodically (on 1st, 3rd, 5th, 7th... responses)
                session.response_count += 1
                if session.response_count % 2 == 1:  # Show on odd response numbers
                    final_content = f"{final_content}\n\n*{DISCLAIMER_TEXT}*"
                
                session.add_message("assistant", final_content)
                print(f"\nAssistant: {final_content}")
                
                # Update session stats
                if hasattr(response, 'usage') and response.usage:
                    session.total_tokens += response.usage.total_tokens
                
            except Exception as e:
                logger.error(f"Chat API call failed: {e}")
                fallback = get_fallback_response()
                print(f"\nAssistant: {fallback}")
                session.add_message("assistant", fallback)
        
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user.")
            logger.info(f"Session {session.session_id} interrupted")
            break
        except Exception as e:
            logger.error(f"Unexpected error in chat loop: {e}", exc_info=True)
            print("\nAssistant: I encountered an unexpected error. Please try again.")

if __name__ == "__main__":
    try:
        enhanced_chat()
    except Exception as e:
        logger.error(f"Application failed to start: {e}", exc_info=True)
        print(f"Failed to start application: {e}")
        print("Please check your API keys and network connection.")