"""Security utilities for input sanitization and response validation."""

import re
import logging
from typing import Dict, Any, List


logger = logging.getLogger(__name__)

# Security configuration
MAX_INPUT_LENGTH = 5000  # Maximum allowed input length


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


def validate_response_security(response: str, function_calls_successful: bool = False) -> str:
    """Validate response content for security and appropriateness.
    
    Args:
        response: The response text to validate
        function_calls_successful: Whether function calls were successful
        
    Returns:
        str: Validated response or security fallback message
    """
    if not isinstance(response, str) or not response.strip():
        return get_fallback_response()
    
    # Enhanced financial keywords - including factor, dividend, ESG terms
    financial_keywords = [
        'benchmark', 'minimum', 'portfolio', 'investment', 'fund', 'etf', 'mutual fund',
        'allocation', 'asset', 'equity', 'bond', 'fixed income', 'international',
        'domestic', 'large cap', 'small cap', 'mid cap', 'growth', 'value', 'blend',
        'sector', 'dividend', 'yield', 'expense ratio', 'tracking error', 'returns',
        'performance', 'volatility', 'risk', 'duration', 'credit', 'currency',
        'emerging markets', 'developed markets', 'real estate', 'commodities',
        'alternatives', 'hedge fund', 'private equity', 'inflation', 'treasury',
        'corporate', 'government', 'municipal', 'tax', 'after-tax', 'pre-tax',
        'gross', 'net', 'fee', 'management', 'advisory', 'fiduciary', 'suitability',
        'risk tolerance', 'time horizon', 'liquidity', 'rebalancing', 'diversification',
        'correlation', 'beta', 'alpha', 'sharpe ratio', 'information ratio',
        'standard deviation', 'var', 'maximum drawdown', 'upside', 'downside',
        'factor', 'tilt', 'momentum', 'quality', 'low volatility', 'size', 'profitability',
        'esg', 'sustainable', 'environmental', 'social', 'governance', 'impact',
        'russell', 'msci', 'sp', 's&p', 'nasdaq', 'ftse', 'dow', 'vanguard',
        'blackrock', 'fidelity', 'schwab', 'ishares', 'spdr', 'invesco',
        'wisdomtree', 'first trust', 'proshares', 'direxion', 'leveraged',
        'inverse', 'short', 'long', 'bull', 'bear', 'market', 'index'
    ]
    
    response_lower = response.lower()
    
    # If function calls were successful, be more lenient with keyword validation
    # Focus on blocking obvious injection attempts rather than keyword requirements
    if function_calls_successful:
        # Only block responses that are clearly inappropriate or injection attempts
        inappropriate_patterns = [
            r'i\s+am\s+(?:not\s+)?(?:a\s+)?(?:an\s+)?assistant',
            r'i\s+(?:can\'?t|cannot|won\'?t|will\s+not)\s+help',
            r'i\s+(?:don\'?t|do\s+not)\s+have\s+access',
            r'as\s+(?:a\s+)?(?:an\s+)?ai\s+language\s+model',
            r'i\'?m\s+(?:just\s+)?(?:a\s+)?(?:an\s+)?ai',
            r'(?:sorry|unfortunately),?\s+i\s+(?:can\'?t|cannot)',
            r'that\s+(?:question\s+)?(?:is\s+)?(?:outside|beyond)\s+my',
            r'(?:contact|speak\s+(?:with|to)|consult)\s+(?:a\s+)?(?:qualified\s+)?(?:financial\s+)?(?:advisor|professional)',
            # System/role leakage patterns
            r'system\s*:',
            r'assistant\s*:',
            r'user\s*:',
            r'\[system\]',
            r'\[assistant\]',
            r'\[user\]'
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, response_lower, re.IGNORECASE):
                logger.warning(f"SECURITY: Blocked inappropriate response pattern: {pattern}")
                return get_fallback_response()
        
        # If function calls succeeded and no inappropriate patterns, allow the response
        return response
    
    # Original validation logic for cases where function calls weren't successful
    has_financial_content = any(keyword in response_lower for keyword in financial_keywords)
    
    if not has_financial_content:
        # Check for common AI assistant phrases that indicate the model is not staying in character
        ai_phrases = [
            'i am an ai', 'i am a language model', 'i am claude', 'i am chatgpt',
            'i am gpt', 'as an ai', 'as a language model', 'as chatgpt', 'as claude',
            'i cannot browse', 'i cannot access', 'i don\'t have access to real-time',
            'i don\'t have the ability to', 'i\'m not able to access',
            'contact a financial advisor', 'consult with a financial advisor',
            'speak to a financial advisor', 'please consult', 'seek professional advice'
        ]
        
        if any(phrase in response_lower for phrase in ai_phrases):
            logger.warning("SECURITY: AI assistant phrases detected, using fallback")
            return get_fallback_response()
        
        # Also check for attempts to break character or change context
        break_character_patterns = [
            r'ignore\s+previous\s+instructions',
            r'forget\s+what\s+i\s+told\s+you',
            r'new\s+conversation',
            r'reset\s+conversation',
            r'you\s+are\s+now',
            r'pretend\s+to\s+be',
            r'act\s+like'
        ]
        
        for pattern in break_character_patterns:
            if re.search(pattern, response_lower, re.IGNORECASE):
                logger.warning(f"SECURITY: Character break attempt detected: {pattern}")
                return get_fallback_response()
        
        logger.warning("SECURITY: Response lacks financial content and context")
        return get_fallback_response()
    
    return response


def get_fallback_response() -> str:
    """Get a random fallback response for security issues."""
    import random
    fallback_responses = [
        "I'm experiencing technical difficulties accessing the benchmark database. Please try again in a moment, or contact your Sales representative for immediate assistance.",
        "I'm unable to process that request right now due to system issues. For urgent benchmark questions, please reach out to your Sales team directly.",
        "The benchmark lookup system is temporarily unavailable. I apologize for the inconvenience. Your Sales representative can provide immediate assistance with benchmark eligibility questions.",
    ]
    return random.choice(fallback_responses)