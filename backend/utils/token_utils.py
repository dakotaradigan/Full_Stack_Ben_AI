"""Token counting and trimming utilities for chat messages."""

import tiktoken
from typing import List, Dict, Any


# Default model configuration
DEFAULT_CHAT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 128000


def num_tokens_from_messages(messages: List[Dict[str, Any]], model: str = DEFAULT_CHAT_MODEL) -> int:
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


def trim_history(messages: List[Dict[str, Any]], limit: int = DEFAULT_MAX_TOKENS - 2000) -> bool:
    """Trim oldest user/assistant pairs until total tokens are under limit.
    
    Returns:
        bool: True if messages were trimmed, False otherwise
    """
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


def estimate_cost(tokens: int, model: str = DEFAULT_CHAT_MODEL) -> float:
    """Estimate cost of API request based on token count."""
    # GPT-4o pricing (as of 2024)
    if model == "gpt-4o":
        return tokens * 0.000005  # $5 per 1M tokens input
    elif model.startswith("gpt-4"):
        return tokens * 0.00003  # $30 per 1M tokens input
    elif model.startswith("gpt-3.5"):
        return tokens * 0.0000015  # $1.5 per 1M tokens input
    else:
        return tokens * 0.000005  # Default to GPT-4o pricing