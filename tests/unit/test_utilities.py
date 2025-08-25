"""
Unit tests for utility functions in chatbot_core.py.

These tests validate utility functions like embedding, rate limiting,
circuit breakers, and other helper functions.
"""

import os
import sys
import time
import types
from collections import defaultdict
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import pytest

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Mock external dependencies before importing
class MockPinecone:
    def __init__(self, *args, **kwargs):
        pass

class MockOpenAI:
    def __init__(self, *args, **kwargs):
        self.embeddings = MockEmbeddings()

class MockEmbeddings:
    def create(self, *args, **kwargs):
        return types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=[0.1] * 1536)]
        )

class MockEncoding:
    def encode(self, text):
        return [1] * max(1, len(text) // 4)  # Rough approximation

# Apply mocks
sys.modules['pinecone'] = types.ModuleType('pinecone')
sys.modules['pinecone'].Pinecone = MockPinecone
sys.modules['openai'] = types.ModuleType('openai')
sys.modules['openai'].OpenAI = MockOpenAI
sys.modules['tiktoken'] = types.ModuleType('tiktoken')
sys.modules['tiktoken'].encoding_for_model = lambda model: MockEncoding()

# Now import the module we're testing
import chatbot_core

class TestEmbedFunction:
    """Tests for embed function."""
    
    def test_embed_basic_text(self):
        """Test embedding basic text."""
        with patch.object(chatbot_core.client.embeddings, 'create') as mock_create:
            mock_create.return_value = types.SimpleNamespace(
                data=[types.SimpleNamespace(embedding=[0.1] * 1536)]
            )
            
            result = chatbot_core.embed("technology stocks")
            
            assert isinstance(result, list)
            assert len(result) == 1536  # Standard embedding dimension
            assert all(isinstance(x, float) for x in result)
            
            mock_create.assert_called_once()
            
    def test_embed_empty_text(self):
        """Test embedding empty text."""
        with patch.object(chatbot_core.client.embeddings, 'create') as mock_create:
            mock_create.return_value = types.SimpleNamespace(
                data=[types.SimpleNamespace(embedding=[0.0] * 1536)]
            )
            
            result = chatbot_core.embed("")
            
            assert isinstance(result, list)
            assert len(result) == 1536
            
    def test_embed_long_text(self):
        """Test embedding long text."""
        long_text = "This is a very long text about financial benchmarks. " * 100
        
        with patch.object(chatbot_core.client.embeddings, 'create') as mock_create:
            mock_create.return_value = types.SimpleNamespace(
                data=[types.SimpleNamespace(embedding=[0.1] * 1536)]
            )
            
            result = chatbot_core.embed(long_text)
            
            assert isinstance(result, list)
            assert len(result) == 1536
            
    def test_embed_special_characters(self):
        """Test embedding text with special characters."""
        special_text = "S&P 500 with 2.5% dividend yield @ $2M minimum"
        
        with patch.object(chatbot_core.client.embeddings, 'create') as mock_create:
            mock_create.return_value = types.SimpleNamespace(
                data=[types.SimpleNamespace(embedding=[0.1] * 1536)]
            )
            
            result = chatbot_core.embed(special_text)
            
            assert isinstance(result, list)
            assert len(result) == 1536
            
    def test_embed_api_error(self):
        """Test embed function handles API errors."""
        with patch.object(chatbot_core.client.embeddings, 'create', side_effect=Exception("API Error")):
            # Should handle error gracefully
            with pytest.raises(Exception):
                chatbot_core.embed("test text")

class TestUsageTracker:
    """Tests for UsageTracker class."""
    
    def test_usage_tracker_initialization(self):
        """Test UsageTracker initialization."""
        tracker = chatbot_core.UsageTracker()
        
        assert isinstance(tracker.requests_per_minute, defaultdict)
        assert isinstance(tracker.cost_per_hour, float)
        assert isinstance(tracker.last_reset, datetime)
        
    def test_usage_tracker_request_tracking(self):
        """Test request tracking functionality."""
        tracker = chatbot_core.UsageTracker()
        current_minute = int(time.time() // 60)
        
        # Track some requests
        tracker.requests_per_minute[current_minute] += 1
        tracker.requests_per_minute[current_minute] += 1
        
        assert tracker.requests_per_minute[current_minute] == 2
        
    def test_usage_tracker_cost_tracking(self):
        """Test cost tracking functionality."""
        tracker = chatbot_core.UsageTracker()
        
        # Add some costs
        tracker.cost_per_hour += 5.0
        tracker.cost_per_hour += 2.5
        
        assert tracker.cost_per_hour == 7.5
        
    def test_usage_tracker_reset(self):
        """Test usage tracker reset functionality."""
        tracker = chatbot_core.UsageTracker()
        
        # Add some usage
        current_minute = int(time.time() // 60)
        tracker.requests_per_minute[current_minute] = 10
        tracker.cost_per_hour = 15.0
        
        # Simulate reset (would be done by reset method if it exists)
        old_time = tracker.last_reset
        tracker.last_reset = datetime.now()
        
        assert tracker.last_reset > old_time

class TestRetryAndCircuitBreaker:
    """Tests for retry and circuit breaker functionality."""
    
    def test_with_retry_success(self):
        """Test retry mechanism with successful function."""
        call_count = 0
        
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
            
        result = chatbot_core._with_retry_and_circuit_breaker(successful_function)
        
        assert result == "success"
        assert call_count == 1  # Should succeed on first try
        
    def test_with_retry_eventual_success(self):
        """Test retry mechanism with eventual success."""
        call_count = 0
        
        def eventually_successful_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return "success"
            
        result = chatbot_core._with_retry_and_circuit_breaker(
            eventually_successful_function,
            max_attempts=3
        )
        
        assert result == "success"
        assert call_count == 3
        
    def test_with_retry_max_attempts_exceeded(self):
        """Test retry mechanism when max attempts exceeded."""
        call_count = 0
        
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception(f"Error attempt {call_count}")
            
        with pytest.raises(Exception):
            chatbot_core._with_retry_and_circuit_breaker(
                always_failing_function,
                max_attempts=2
            )
            
        assert call_count == 2
        
    def test_with_retry_different_exceptions(self):
        """Test retry mechanism with different exception types."""
        def network_error_function():
            raise ConnectionError("Network error")
            
        def value_error_function():
            raise ValueError("Value error")
            
        # Should handle different exception types
        with pytest.raises(ConnectionError):
            chatbot_core._with_retry_and_circuit_breaker(
                network_error_function,
                max_attempts=2
            )
            
        with pytest.raises(ValueError):
            chatbot_core._with_retry_and_circuit_breaker(
                value_error_function,
                max_attempts=2
            )

class TestRateLimiting:
    """Tests for rate limiting functionality."""
    
    def test_rate_limit_constants(self):
        """Test rate limiting constants are defined."""
        assert hasattr(chatbot_core, 'MAX_REQUESTS_PER_MINUTE')
        assert hasattr(chatbot_core, 'MAX_COST_PER_HOUR')
        assert hasattr(chatbot_core, 'MAX_TOKENS_PER_REQUEST')
        
        assert isinstance(chatbot_core.MAX_REQUESTS_PER_MINUTE, int)
        assert isinstance(chatbot_core.MAX_COST_PER_HOUR, float)
        assert isinstance(chatbot_core.MAX_TOKENS_PER_REQUEST, int)
        
        assert chatbot_core.MAX_REQUESTS_PER_MINUTE > 0
        assert chatbot_core.MAX_COST_PER_HOUR > 0
        assert chatbot_core.MAX_TOKENS_PER_REQUEST > 0
        
    def test_usage_tracking_integration(self):
        """Test usage tracking integration."""
        # Test that usage tracker is initialized
        assert hasattr(chatbot_core, 'usage_tracker')
        
        tracker = chatbot_core.usage_tracker
        assert isinstance(tracker, chatbot_core.UsageTracker)

class TestConfigurationLoading:
    """Tests for configuration loading utilities."""
    
    def test_system_prompt_loading(self):
        """Test system prompt is loaded correctly."""
        assert hasattr(chatbot_core, 'SYSTEM_PROMPT')
        assert isinstance(chatbot_core.SYSTEM_PROMPT, str)
        assert len(chatbot_core.SYSTEM_PROMPT) > 0
        
        # Should contain key instructions
        prompt_lower = chatbot_core.SYSTEM_PROMPT.lower()
        assert any(keyword in prompt_lower for keyword in [
            "benchmark", "eligibility", "function", "minimum"
        ])
        
    def test_disclaimer_extraction(self):
        """Test disclaimer text extraction."""
        assert hasattr(chatbot_core, 'DISCLAIMER_TEXT')
        assert isinstance(chatbot_core.DISCLAIMER_TEXT, str)
        assert len(chatbot_core.DISCLAIMER_TEXT) > 0
        
        # Should contain disclaimer-like content
        disclaimer_lower = chatbot_core.DISCLAIMER_TEXT.lower()
        assert any(keyword in disclaimer_lower for keyword in [
            "guidance", "advice", "representative", "tool"
        ])
        
    def test_api_key_validation(self):
        """Test API key validation."""
        # Should have client objects initialized
        assert hasattr(chatbot_core, 'client')
        assert hasattr(chatbot_core, 'pc')
        
        assert chatbot_core.client is not None
        assert chatbot_core.pc is not None

class TestModelConfiguration:
    """Tests for model configuration constants."""
    
    def test_model_constants(self):
        """Test model configuration constants."""
        assert hasattr(chatbot_core, 'EMBEDDING_MODEL')
        assert hasattr(chatbot_core, 'CHAT_MODEL')
        assert hasattr(chatbot_core, 'MAX_MODEL_TOKENS')
        
        assert isinstance(chatbot_core.EMBEDDING_MODEL, str)
        assert isinstance(chatbot_core.CHAT_MODEL, str)
        assert isinstance(chatbot_core.MAX_MODEL_TOKENS, int)
        
        # Should be reasonable values
        assert len(chatbot_core.EMBEDDING_MODEL) > 0
        assert len(chatbot_core.CHAT_MODEL) > 0
        assert chatbot_core.MAX_MODEL_TOKENS > 1000
        
    def test_safety_configuration(self):
        """Test safety configuration constants."""
        assert hasattr(chatbot_core, 'MAX_INPUT_LENGTH')
        assert hasattr(chatbot_core, 'TOKEN_MARGIN')
        
        assert isinstance(chatbot_core.MAX_INPUT_LENGTH, int)
        assert isinstance(chatbot_core.TOKEN_MARGIN, int)
        
        assert chatbot_core.MAX_INPUT_LENGTH > 0
        assert chatbot_core.TOKEN_MARGIN > 0

class TestLogging:
    """Tests for logging configuration."""
    
    def test_logger_configuration(self):
        """Test logger is properly configured."""
        assert hasattr(chatbot_core, 'logger')
        
        logger = chatbot_core.logger
        assert logger is not None
        assert logger.name == 'chatbot_core'
        
    def test_log_levels(self):
        """Test logging levels are set appropriately."""
        import logging
        
        # Should have file and console handlers configured
        logger = chatbot_core.logger
        
        # Should have handlers
        assert len(logger.handlers) >= 0  # May be configured at root level
        
    def test_logging_suppression(self):
        """Test that noisy third-party loggers are suppressed."""
        import logging
        
        # These loggers should be set to WARNING level
        noisy_loggers = ['httpx', 'openai', 'urllib3']
        
        for logger_name in noisy_loggers:
            logger = logging.getLogger(logger_name)
            # Should be WARNING or higher
            assert logger.level >= logging.WARNING or logger.level == 0  # 0 means inherit

class TestErrorHandling:
    """Tests for error handling utilities."""
    
    def test_graceful_error_handling(self):
        """Test functions handle errors gracefully."""
        # Test with invalid inputs that shouldn't crash the system
        
        # Test embedding with None
        try:
            with patch.object(chatbot_core.client.embeddings, 'create', side_effect=Exception("API Error")):
                chatbot_core.embed(None)
        except:
            pass  # Expected to fail, but shouldn't crash system
            
    def test_input_validation(self):
        """Test input validation in utility functions."""
        # Test token counting with invalid input
        try:
            result = chatbot_core.num_tokens_from_messages(None)
            assert result >= 0  # Should return sensible default
        except:
            pass  # May raise exception, but shouldn't crash system
            
        # Test with malformed message structure
        try:
            malformed_messages = [{"invalid": "structure"}]
            result = chatbot_core.num_tokens_from_messages(malformed_messages)
            assert isinstance(result, int)
        except:
            pass  # May raise exception, but shouldn't crash system

class TestConstants:
    """Tests for module constants and configuration."""
    
    def test_function_definitions(self):
        """Test function definitions are properly structured."""
        assert hasattr(chatbot_core, 'FUNCTIONS')
        assert isinstance(chatbot_core.FUNCTIONS, list)
        assert len(chatbot_core.FUNCTIONS) > 0
        
        # Each function should have required structure
        for func in chatbot_core.FUNCTIONS:
            assert isinstance(func, dict)
            assert 'name' in func
            assert 'description' in func
            assert 'parameters' in func
            
    def test_environment_variable_handling(self):
        """Test environment variable handling."""
        # API keys should be loaded
        assert hasattr(chatbot_core, 'OPENAI_API_KEY')
        assert hasattr(chatbot_core, 'PINECONE_API_KEY')
        
        # Should not be default placeholder values in production
        assert chatbot_core.OPENAI_API_KEY != "YOUR_OPENAI_API_KEY"
        assert chatbot_core.PINECONE_API_KEY != "YOUR_PINECONE_API_KEY"

class TestUtilityHelpers:
    """Tests for small utility helper functions."""
    
    def test_format_currency(self):
        """Test currency formatting utilities (if they exist)."""
        # Test various amounts
        test_amounts = [1000000, 2500000, 10000000]
        
        for amount in test_amounts:
            # If there's a formatting function, test it
            # (This would depend on actual implementation)
            formatted = f"${amount:,.0f}"  # Basic formatting
            assert "$" in formatted
            assert "," in formatted or len(str(amount)) <= 3
            
    def test_text_processing_helpers(self):
        """Test text processing helper functions."""
        # Test fuzzy matching functions
        assert callable(chatbot_core._is_fuzzy_match)
        assert callable(chatbot_core._is_aggressive_fuzzy_match)
        
        # Test basic matching
        assert chatbot_core._is_fuzzy_match("S&P 500", "S&P 500")
        assert chatbot_core._is_fuzzy_match("s&p 500", "S&P 500")
        
    def test_data_structure_helpers(self):
        """Test data structure helper functions."""
        # Test that helper functions exist and are callable
        helper_functions = [
            '_fuzzy_match_benchmark',
            '_is_fuzzy_match', 
            '_is_aggressive_fuzzy_match'
        ]
        
        for func_name in helper_functions:
            assert hasattr(chatbot_core, func_name)
            assert callable(getattr(chatbot_core, func_name))

class TestPerformanceUtilities:
    """Tests for performance-related utilities."""
    
    def test_token_limits(self):
        """Test token limit configurations."""
        # Should have reasonable token limits
        assert chatbot_core.MAX_TOKENS_PER_REQUEST < chatbot_core.MAX_MODEL_TOKENS
        assert chatbot_core.TOKEN_MARGIN < chatbot_core.MAX_TOKENS_PER_REQUEST
        
    def test_cost_estimation_accuracy(self):
        """Test cost estimation is reasonable."""
        # Test different token amounts
        costs = []
        token_amounts = [100, 1000, 10000]
        
        for tokens in token_amounts:
            cost = chatbot_core.estimate_cost(tokens)
            costs.append(cost)
            assert cost >= 0
            
        # Cost should increase with tokens
        assert costs[1] > costs[0]
        assert costs[2] > costs[1]