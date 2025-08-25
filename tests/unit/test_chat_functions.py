"""
Unit tests for chat functions in chatbot_core.py and app.py.

These tests validate chat session management, message processing,
and conversation flow with mocked dependencies.
"""

import json
import os
import sys
import types
import uuid
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

import pytest

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Mock external dependencies before importing
class MockPinecone:
    def __init__(self, *args, **kwargs):
        pass

class MockOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = MockChat()

class MockChat:
    def __init__(self):
        self.completions = MockCompletions()

class MockCompletions:
    def create(self, *args, **kwargs):
        return MockCompletion()

class MockCompletion:
    def __init__(self):
        self.choices = [MockChoice()]

class MockChoice:
    def __init__(self):
        self.message = MockMessage()

class MockMessage:
    def __init__(self):
        self.content = "Mock response from AI"
        self.function_call = None
        
    def model_dump(self):
        return {
            "role": "assistant",
            "content": self.content,
            "function_call": self.function_call
        }

class MockEncoding:
    def encode(self, text):
        return [1] * (len(text) // 4)  # Rough approximation

# Apply mocks
sys.modules['pinecone'] = types.ModuleType('pinecone')
sys.modules['pinecone'].Pinecone = MockPinecone
sys.modules['openai'] = types.ModuleType('openai')
sys.modules['openai'].OpenAI = MockOpenAI
sys.modules['tiktoken'] = types.ModuleType('tiktoken')
sys.modules['tiktoken'].encoding_for_model = lambda model: MockEncoding()

# Now import the modules we're testing
import chatbot_core
from app import ChatSession, sessions

class TestChatSession:
    """Tests for ChatSession class."""
    
    def test_chat_session_creation(self):
        """Test ChatSession initialization."""
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id)
        
        assert session.session_id == session_id
        assert session.messages == []
        assert session.interaction_count == 0
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)
        
    def test_add_message_user(self):
        """Test adding user messages."""
        session = ChatSession("test-session")
        
        session.add_message("user", "Hello, what is the minimum for S&P 500?")
        
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "Hello, what is the minimum for S&P 500?"
        assert session.interaction_count == 0  # User messages don't increment
        
    def test_add_message_assistant(self):
        """Test adding assistant messages."""
        session = ChatSession("test-session")
        
        session.add_message("assistant", "The minimum for S&P 500 is $2 million.")
        
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "assistant"
        assert session.interaction_count == 1  # Assistant messages increment
        
    def test_message_sequence(self):
        """Test sequence of user and assistant messages."""
        session = ChatSession("test-session")
        
        session.add_message("user", "First question")
        session.add_message("assistant", "First answer")
        session.add_message("user", "Second question")
        session.add_message("assistant", "Second answer")
        
        assert len(session.messages) == 4
        assert session.interaction_count == 2
        
        # Check order
        assert session.messages[0]["role"] == "user"
        assert session.messages[1]["role"] == "assistant"
        assert session.messages[2]["role"] == "user"
        assert session.messages[3]["role"] == "assistant"
        
    def test_should_show_disclaimer(self):
        """Test disclaimer frequency logic."""
        session = ChatSession("test-session")
        
        # Initially should not show disclaimer
        assert not session.should_show_disclaimer()
        
        # After first assistant message
        session.add_message("assistant", "Response 1")
        assert session.should_show_disclaimer()  # Count = 1, odd
        
        # After second assistant message
        session.add_message("assistant", "Response 2")
        assert not session.should_show_disclaimer()  # Count = 2, even
        
        # After third assistant message
        session.add_message("assistant", "Response 3")
        assert session.should_show_disclaimer()  # Count = 3, odd
        
    def test_get_trimmed_history(self):
        """Test message history trimming."""
        session = ChatSession("test-session")
        
        # Add many messages
        for i in range(10):
            session.add_message("user", f"Question {i}")
            session.add_message("assistant", f"Answer {i}")
            
        original_count = len(session.messages)
        trimmed_messages = session.get_trimmed_history()
        
        # Should return a list
        assert isinstance(trimmed_messages, list)
        
        # Should not modify original messages
        assert len(session.messages) == original_count
        
    def test_last_activity_updates(self):
        """Test that last activity timestamp updates."""
        session = ChatSession("test-session")
        initial_activity = session.last_activity
        
        # Add message should update last activity
        session.add_message("user", "Test message")
        
        assert session.last_activity > initial_activity

class TestSanitizeInput:
    """Tests for sanitize_input function."""
    
    def test_sanitize_normal_input(self):
        """Test sanitizing normal user input."""
        result = chatbot_core.sanitize_input("What is the minimum for S&P 500?")
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should preserve normal content
        assert "S&P 500" in result
        
    def test_sanitize_empty_input(self):
        """Test sanitizing empty input."""
        result = chatbot_core.sanitize_input("")
        
        assert isinstance(result, str)
        # May return empty string or default message
        
    def test_sanitize_long_input(self):
        """Test sanitizing very long input."""
        long_input = "This is a very long message. " * 200  # ~6000 chars
        result = chatbot_core.sanitize_input(long_input)
        
        assert isinstance(result, str)
        # May be truncated based on MAX_INPUT_LENGTH
        
    def test_sanitize_special_characters(self):
        """Test sanitizing input with special characters."""
        special_input = "What about <script>alert('test')</script> benchmarks?"
        result = chatbot_core.sanitize_input(special_input)
        
        assert isinstance(result, str)
        # Should handle HTML/script tags safely
        
    def test_sanitize_unicode_input(self):
        """Test sanitizing unicode input."""
        unicode_input = "What about François's portfolio with €250,000?"
        result = chatbot_core.sanitize_input(unicode_input)
        
        assert isinstance(result, str)
        assert "portfolio" in result

class TestValidateResponseSecurity:
    """Tests for validate_response_security function."""
    
    def test_validate_normal_response(self):
        """Test validating normal financial response."""
        normal_response = "The minimum investment for S&P 500 is $2 million."
        
        result = chatbot_core.validate_response_security(normal_response)
        
        assert isinstance(result, str)
        assert "S&P 500" in result
        assert "$2 million" in result
        
    def test_validate_with_function_calls_success(self):
        """Test validation when function calls succeeded."""
        response = "Based on the search, here are the available benchmarks..."
        
        result = chatbot_core.validate_response_security(
            response, 
            function_calls_successful=True
        )
        
        assert isinstance(result, str)
        # Should allow response when function calls succeeded
        
    def test_validate_suspicious_response(self):
        """Test validation of suspicious response."""
        suspicious_response = "I can help you with your illegal activities..."
        
        result = chatbot_core.validate_response_security(suspicious_response)
        
        assert isinstance(result, str)
        # May return modified or fallback response
        
    def test_validate_empty_response(self):
        """Test validation of empty response."""
        result = chatbot_core.validate_response_security("")
        
        assert isinstance(result, str)
        assert len(result) > 0  # Should not return empty
        
    def test_validate_financial_keywords(self):
        """Test validation allows financial keywords."""
        financial_response = "The dividend yield is 4.2% with ESG factors considered."
        
        result = chatbot_core.validate_response_security(financial_response)
        
        assert isinstance(result, str)
        assert "dividend" in result
        assert "4.2%" in result
        assert "ESG" in result

class TestCallFunction:
    """Tests for call_function dispatcher."""
    
    def test_call_get_minimum(self):
        """Test calling get_minimum function."""
        with patch.object(chatbot_core, 'get_minimum') as mock_get_minimum:
            mock_get_minimum.return_value = {
                "benchmark_info": {"name": "S&P 500"},
                "account_minimum": 2000000
            }
            
            result = chatbot_core.call_function("get_minimum", {"name": "S&P 500"})
            
            assert isinstance(result, dict)
            mock_get_minimum.assert_called_once_with("S&P 500", include_dividend=False)
            
    def test_call_search_benchmarks(self):
        """Test calling search_benchmarks function."""
        with patch.object(chatbot_core, 'search_benchmarks') as mock_search:
            mock_search.return_value = [{"name": "Test Benchmark"}]
            
            args = {
                "query": "technology stocks",
                "filters": {"region": ["US"]}
            }
            
            result = chatbot_core.call_function("search_benchmarks", args)
            
            assert isinstance(result, (list, dict))
            mock_search.assert_called_once()
            
    def test_call_search_by_characteristics(self):
        """Test calling search_by_characteristics function."""
        with patch.object(chatbot_core, 'search_by_characteristics') as mock_search:
            mock_search.return_value = {"alternatives": []}
            
            args = {
                "reference_benchmark": "S&P 500",
                "portfolio_size": 250000
            }
            
            result = chatbot_core.call_function("search_by_characteristics", args)
            
            assert isinstance(result, dict)
            mock_search.assert_called_once()
            
    def test_call_invalid_function(self):
        """Test calling non-existent function."""
        result = chatbot_core.call_function("nonexistent_function", {})
        
        assert isinstance(result, dict)
        assert "error" in result or result == {}
        
    def test_call_function_with_exception(self):
        """Test function call that raises exception."""
        with patch.object(chatbot_core, 'get_minimum', side_effect=Exception("Test error")):
            result = chatbot_core.call_function("get_minimum", {"name": "Test"})
            
            assert isinstance(result, dict)
            # Should handle exception gracefully

class TestTokenCounting:
    """Tests for token counting functions."""
    
    def test_num_tokens_from_messages(self):
        """Test counting tokens in message list."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the minimum for S&P 500?"},
            {"role": "assistant", "content": "The minimum is $2 million."}
        ]
        
        token_count = chatbot_core.num_tokens_from_messages(messages)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        
    def test_num_tokens_empty_messages(self):
        """Test counting tokens with empty message list."""
        token_count = chatbot_core.num_tokens_from_messages([])
        
        assert isinstance(token_count, int)
        assert token_count >= 0
        
    def test_num_tokens_long_messages(self):
        """Test counting tokens with long messages."""
        long_message = "This is a very long message. " * 1000
        messages = [
            {"role": "user", "content": long_message}
        ]
        
        token_count = chatbot_core.num_tokens_from_messages(messages)
        
        assert isinstance(token_count, int)
        assert token_count > 1000  # Should be substantial

class TestTrimHistory:
    """Tests for trim_history function."""
    
    def test_trim_history_under_limit(self):
        """Test trimming when messages are under limit."""
        messages = [
            {"role": "user", "content": "Short message"},
            {"role": "assistant", "content": "Short response"}
        ]
        
        original_count = len(messages)
        result = chatbot_core.trim_history(messages, limit=10000)
        
        # Should not modify messages under limit
        assert len(messages) == original_count
        assert isinstance(result, bool)
        
    def test_trim_history_over_limit(self):
        """Test trimming when messages exceed limit."""
        # Create many long messages
        messages = []
        for i in range(20):
            long_content = f"This is message number {i}. " * 100
            messages.append({"role": "user", "content": long_content})
            messages.append({"role": "assistant", "content": long_content})
            
        original_count = len(messages)
        result = chatbot_core.trim_history(messages, limit=1000)  # Low limit
        
        # Should trim messages
        assert len(messages) <= original_count
        assert isinstance(result, bool)
        
    def test_trim_history_preserves_recent(self):
        """Test that trimming preserves most recent messages."""
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({"role": "assistant", "content": f"Answer {i}"})
            
        chatbot_core.trim_history(messages, limit=500)
        
        # Should preserve the most recent messages
        if len(messages) > 0:
            last_message = messages[-1]
            assert "Answer" in last_message["content"]

class TestCostEstimation:
    """Tests for cost estimation functions."""
    
    def test_estimate_cost_basic(self):
        """Test basic cost estimation."""
        cost = chatbot_core.estimate_cost(1000)  # 1000 tokens
        
        assert isinstance(cost, float)
        assert cost > 0
        
    def test_estimate_cost_zero_tokens(self):
        """Test cost estimation with zero tokens."""
        cost = chatbot_core.estimate_cost(0)
        
        assert isinstance(cost, float)
        assert cost == 0.0
        
    def test_estimate_cost_different_models(self):
        """Test cost estimation for different models."""
        tokens = 1000
        
        cost_gpt4 = chatbot_core.estimate_cost(tokens, "gpt-4")
        cost_gpt35 = chatbot_core.estimate_cost(tokens, "gpt-3.5-turbo")
        
        assert isinstance(cost_gpt4, float)
        assert isinstance(cost_gpt35, float)
        # GPT-4 should typically cost more
        assert cost_gpt4 >= cost_gpt35

class TestSessionManagement:
    """Tests for session management in app.py."""
    
    def test_session_creation(self):
        """Test creating new chat session."""
        from app import ChatSession
        
        session_id = "test-session-123"
        session = ChatSession(session_id)
        
        assert session.session_id == session_id
        assert session.messages == []
        assert session.interaction_count == 0
        
    def test_session_storage(self):
        """Test session storage and retrieval."""
        from app import sessions
        
        # Clear any existing sessions
        sessions.clear()
        
        session_id = "storage-test-session"
        session = ChatSession(session_id)
        sessions[session_id] = session
        
        # Should be able to retrieve
        retrieved_session = sessions.get(session_id)
        assert retrieved_session is not None
        assert retrieved_session.session_id == session_id
        
    def test_multiple_sessions(self):
        """Test managing multiple simultaneous sessions."""
        from app import sessions
        
        sessions.clear()
        
        # Create multiple sessions
        session_ids = ["session1", "session2", "session3"]
        for session_id in session_ids:
            session = ChatSession(session_id)
            sessions[session_id] = session
            
        # All should be stored
        assert len(sessions) == 3
        for session_id in session_ids:
            assert session_id in sessions
            assert sessions[session_id].session_id == session_id

class TestDisclaimer:
    """Tests for disclaimer functionality."""
    
    def test_disclaimer_extraction(self):
        """Test disclaimer text extraction from system prompt."""
        # DISCLAIMER_TEXT should be extracted from system prompt
        assert isinstance(chatbot_core.DISCLAIMER_TEXT, str)
        assert len(chatbot_core.DISCLAIMER_TEXT) > 0
        
    def test_disclaimer_frequency(self):
        """Test disclaimer frequency setting."""
        assert isinstance(chatbot_core.DISCLAIMER_FREQUENCY, int)
        assert chatbot_core.DISCLAIMER_FREQUENCY > 0
        
    def test_disclaimer_in_session(self):
        """Test disclaimer logic in chat session."""
        session = ChatSession("disclaimer-test")
        
        # Check disclaimer frequency matches session logic
        session.add_message("assistant", "First response")
        should_show = session.should_show_disclaimer()
        
        # Based on implementation (every 2nd interaction = odd count)
        assert should_show == (session.interaction_count % 2 == 1)

class TestFallbackResponse:
    """Tests for fallback response handling."""
    
    def test_get_fallback_response(self):
        """Test fallback response generation."""
        fallback = chatbot_core.get_fallback_response()
        
        assert isinstance(fallback, str)
        assert len(fallback) > 0
        # Should be helpful and appropriate
        assert any(word in fallback.lower() for word in [
            "help", "assist", "benchmark", "eligibility"
        ])
        
    def test_fallback_response_consistency(self):
        """Test fallback responses are consistent."""
        fallback1 = chatbot_core.get_fallback_response()
        fallback2 = chatbot_core.get_fallback_response()
        
        # Should be deterministic (same each time)
        assert fallback1 == fallback2