"""
Bridge layer for chatbot_core.py maintaining backward compatibility while using new services.
This ensures that app.py and other modules can continue importing from chatbot_core without changes.
"""

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

from utils.description_utils import build_semantic_description

# Import new services
from services.search_service import SearchService
from services.benchmark_service import BenchmarkService
from services.chat_service import ChatService
from models.chat_models import UsageTracker, ChatSession, CircuitBreaker

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

# Load configuration from files
with open("config/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# Extract disclaimer text
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

# Maintain original client for backward compatibility
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Configuration constants
EMBEDDING_MODEL = "text-embedding-3-small" 
CHAT_MODEL = "gpt-4o"
MAX_MODEL_TOKENS = 128000
TOKEN_MARGIN = 1000

# Security and Safety Configuration
MAX_INPUT_LENGTH = 5000
MAX_TOKENS_PER_REQUEST = 12000
MAX_REQUESTS_PER_MINUTE = 100
MAX_COST_PER_HOUR = 50.0

# Initialize services
_search_service = None
_benchmark_service = None
_chat_service = None

def _get_search_service():
    """Get or create search service instance."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service

def _get_benchmark_service():
    """Get or create benchmark service instance."""
    global _benchmark_service
    if _benchmark_service is None:
        _benchmark_service = BenchmarkService()
    return _benchmark_service

def _get_chat_service():
    """Get or create chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service

# Initialize Pinecone index (for backward compatibility)
try:
    INDEX_NAME = "benchmark-index"  # Updated to match existing index name
    if INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"Pinecone index '{INDEX_NAME}' does not exist. Run build_index.py first.")
    index = pc.Index(INDEX_NAME)
    
    # Load benchmark data for the map (backward compatibility)
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

# Circuit breakers for backward compatibility
usage_tracker = UsageTracker()
pinecone_circuit_breaker = CircuitBreaker()

# Function definitions for OpenAI
FUNCTIONS = [
    {
        "name": "search_benchmarks",
        "description": "Semantic search for complex criteria and natural language queries with optional portfolio size filtering",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 5},
                "filters": {"type": "object", "description": "Optional filters"},
                "include_dividend": {"type": "boolean", "default": False},
                "portfolio_size": {"type": "number", "description": "Optional portfolio size for filtering viable benchmarks"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_minimum",
        "description": "Get minimum investment amount for a specific benchmark",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Exact benchmark name"},
                "include_dividend": {"type": "boolean", "default": False},
            },
            "required": ["name"],
        },
    },
    {
        "name": "search_by_characteristics",
        "description": "Find alternatives with similar characteristics to a SPECIFIC NAMED benchmark (e.g., 'Russell 2000', 'MSCI World'). DO NOT use with generic terms like 'small cap' or 'technology'",
        "parameters": {
            "type": "object",
            "properties": {
                "reference_benchmark": {"type": "string", "description": "Must be an actual benchmark name, NOT a generic category"},
                "portfolio_size": {"type": "number"},
                "top_k": {"type": "integer", "default": 5},
                "include_dividend": {"type": "boolean", "default": False},
            },
            "required": ["reference_benchmark"],
        },
    },
    {
        "name": "get_all_benchmarks",
        "description": "Get all benchmarks with optional filtering",
        "parameters": {
            "type": "object",
            "properties": {
                "include_dividend": {"type": "boolean", "default": False},
                "filters": {"type": "object"},
            },
        },
    },
    {
        "name": "blend_minimum",
        "description": "Calculate blended minimum investment for a mix of benchmarks (e.g., 60% MSCI World + 40% MSCI EAFE)",
        "parameters": {
            "type": "object",
            "properties": {
                "allocations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Benchmark name"},
                            "percentage": {"type": "number", "description": "Allocation percentage (0-100)"}
                        },
                        "required": ["name", "percentage"]
                    },
                    "description": "List of benchmarks and their allocations"
                },
                "include_dividend": {"type": "boolean", "default": False}
            },
            "required": ["allocations"]
        }
    },
]

# ============================================================================
# BRIDGE FUNCTIONS - Maintain backward compatibility by delegating to services
# ============================================================================

def sanitize_input(user_input: str) -> str:
    """Sanitize user input to prevent prompt injection attacks."""
    from utils.security_utils import sanitize_input as _sanitize_input
    return _sanitize_input(user_input)

def validate_response_security(response: str, function_calls_successful: bool = False) -> str:
    """Validate response content for security."""
    from utils.security_utils import validate_response_security as _validate_response_security
    return _validate_response_security(response, function_calls_successful)

def num_tokens_from_messages(messages: List[Dict[str, Any]], model: str = CHAT_MODEL) -> int:
    """Return total token count for a list of chat messages."""
    from utils.token_utils import num_tokens_from_messages as _num_tokens_from_messages
    return _num_tokens_from_messages(messages, model)

def trim_history(messages: List[Dict[str, Any]], limit: int = MAX_MODEL_TOKENS - 2000) -> bool:
    """Trim oldest user/assistant pairs until total tokens are under limit."""
    from utils.token_utils import trim_history as _trim_history
    return _trim_history(messages, limit)

def estimate_cost(tokens: int, model: str = CHAT_MODEL) -> float:
    """Estimate cost of API request based on token count."""
    from utils.token_utils import estimate_cost as _estimate_cost
    return _estimate_cost(tokens, model)

def embed(text: str) -> List[float]:
    """Create embeddings with error handling and circuit breaker."""
    return _get_search_service().embed(text)

def get_benchmark(name: str) -> Optional[Dict[str, Any]]:
    """Return benchmark data by name using exact match or fuzzy matching."""
    return _get_benchmark_service().get_benchmark(name)

def get_minimum(name: str, include_dividend: bool = False) -> Dict[str, Any]:
    """Get minimum investment amount for a benchmark."""
    result = _get_benchmark_service().get_minimum(name, include_dividend)
    # Convert to dictionary format for backward compatibility
    return result.to_dict()

def blend_minimum(allocations: List[Dict[str, Any]], include_dividend: bool = False) -> Dict[str, Any]:
    """Calculate blended minimum investment amount."""
    try:
        # Convert percentage to weight if needed (handle both formats)
        converted_allocations = []
        for allocation in allocations:
            converted = allocation.copy()
            if "percentage" in allocation and "weight" not in allocation:
                # Convert percentage (0-100) to weight (0-1)
                converted["weight"] = allocation["percentage"] / 100.0
            elif "weight" not in allocation:
                raise ValueError("Allocation must have either 'percentage' or 'weight' field")
            converted_allocations.append(converted)
        
        result = _get_benchmark_service().blend_minimum(converted_allocations, include_dividend)
        return result.to_dict()
    except Exception as e:
        return {"error": str(e)}

def search_benchmarks(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    include_dividend: bool = False,
    portfolio_size: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Search benchmarks with enhanced error handling and optional portfolio size filtering."""
    return _get_search_service().search_benchmarks(query, top_k, filters, include_dividend, portfolio_size)

def search_by_characteristics(
    reference_benchmark: str,
    portfolio_size: Optional[float] = None,
    top_k: int = 5,
    include_dividend: bool = False
) -> List[Dict[str, Any]]:
    """Search by characteristics with enhanced error handling."""
    return _get_search_service().search_by_characteristics(
        reference_benchmark, portfolio_size, top_k, include_dividend, 
        get_benchmark_func=get_benchmark
    )

def get_all_benchmarks(include_dividend: bool = False, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get all benchmarks with optional filtering."""
    return _get_benchmark_service().get_all_benchmarks(include_dividend, filters)

def call_function(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced function calling with comprehensive error handling."""
    # Create function registry for the chat service
    function_registry = {
        "search_benchmarks": lambda args: {"results": search_benchmarks(**args)},
        "get_minimum": lambda args: get_minimum(**args),
        "blend_minimum": lambda args: blend_minimum(**args),
        "search_by_characteristics": lambda args: {"results": search_by_characteristics(**args)},
        "get_all_benchmarks": lambda args: get_all_benchmarks(**args),
    }
    
    return _get_chat_service().call_function(name, arguments, function_registry)

def get_fallback_response() -> str:
    """Get a random fallback response."""
    return _get_chat_service()._get_fallback_response()

# Additional backward compatibility functions that may be needed
def search_viable_alternatives(
    query: str,
    portfolio_size: float,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    include_dividend: bool = False,
    max_iterations: int = 3
) -> List[Dict[str, Any]]:
    """Search for viable alternatives with portfolio size constraints."""
    return _get_search_service()._search_viable_alternatives(
        query, portfolio_size, top_k, filters, include_dividend, get_benchmark
    )

# Keep any other classes or functions that might be imported directly
# from the original chatbot_core.py for maximum compatibility

def enhanced_chat():
    """Enhanced chat function with comprehensive safety features."""
    # This function can remain as-is or be updated to use the new services
    # For now, keeping the original implementation for compatibility
    session = ChatSession(session_id=f"session_{int(time.time())}")
    session.add_message("system", SYSTEM_PROMPT)
    
    print("🛡️  Enhanced Benchmark Eligibility Assistant (Testing Mode)")
    print("Hi There! I'm here to assist with benchmark eligibility questions. How can I help you today?")
    print(f"Session ID: {session.session_id}")
    print("💡 Testing Mode")  
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("\nUser: ")
            
            # Handle exit commands
            if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
                print("Sad to see ya go! Thanks for using the Assistant.")
                logger.info(f"Session {session.session_id} ended normally - "
                           f"Tokens: {session.total_tokens}, Duration: {session.get_age_minutes():.1f}min")
                break
            
            # Process using new chat service
            chat_service = _get_chat_service()
            function_registry = {
                "search_benchmarks": lambda args: {"results": search_benchmarks(**args)},
                "get_minimum": lambda args: get_minimum(**args),
                "blend_minimum": lambda args: blend_minimum(**args),
                "search_by_characteristics": lambda args: {"results": search_by_characteristics(**args)},
                "get_all_benchmarks": lambda args: get_all_benchmarks(**args),
            }
            
            response_data = chat_service.process_message(session, user_input, function_registry)
            
            if "error" in response_data:
                print(f"\n❌ Error: {response_data['error']}")
            else:
                print(f"\n🤖 Assistant: {response_data['message']}")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            logger.error(f"Chat error: {e}")
            print(f"\n❌ Sorry, an error occurred: {e}")

# Expose all the classes and functions that might be imported
__all__ = [
    # Configuration constants
    'SYSTEM_PROMPT', 'DISCLAIMER_TEXT', 'DISCLAIMER_FREQUENCY',
    'client', 'pc', 'index', 'CHAT_MODEL', 'EMBEDDING_MODEL',
    'MAX_TOKENS_PER_REQUEST', 'MAX_MODEL_TOKENS', 'FUNCTIONS',
    'DATA', 'BENCHMARK_MAP', 'usage_tracker', 'pinecone_circuit_breaker',
    
    # Bridge functions
    'sanitize_input', 'validate_response_security', 'num_tokens_from_messages',
    'trim_history', 'estimate_cost', 'embed', 'get_benchmark', 'get_minimum',
    'blend_minimum', 'search_benchmarks', 'search_by_characteristics',
    'get_all_benchmarks', 'call_function', 'get_fallback_response',
    'search_viable_alternatives', 'enhanced_chat',
    
    # Classes
    'UsageTracker', 'ChatSession', 'CircuitBreaker',
]