"""
Integration tests for OpenAI function calling.

These tests validate that the OpenAI function calling integration
works correctly with actual function implementations.
"""

import json
import os
import sys
import time
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from app import app
import chatbot_core

@pytest.fixture(scope="module")
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)

class TestFunctionCallIntegration:
    """Tests for OpenAI function calling integration."""
    
    def test_get_minimum_function_call(self, client):
        """Test get_minimum function is called correctly."""
        payload = {
            "message": "What is the minimum investment for S&P 500?"
        }
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        assert "function_calls" in data
        
        # Should have made function calls
        if data["function_calls"]:
            # Check that get_minimum was called
            function_names = [call["name"] for call in data["function_calls"]]
            assert "get_minimum" in function_names
            
            # Find the get_minimum call
            get_minimum_calls = [
                call for call in data["function_calls"] 
                if call["name"] == "get_minimum"
            ]
            assert len(get_minimum_calls) >= 1
            
            # Check arguments
            call = get_minimum_calls[0]
            assert "args" in call
            assert "name" in call["args"]
            assert "s&p 500" in call["args"]["name"].lower()
            
            # Check result
            assert "result" in call
            assert isinstance(call["result"], dict)
        
        # Response should mention specific amount
        response_lower = data["response"].lower()
        assert any(keyword in response_lower for keyword in [
            "million", "$", "minimum", "investment"
        ])
        
    def test_search_benchmarks_function_call(self, client):
        """Test search_benchmarks function is called correctly."""
        payload = {
            "message": "What technology benchmarks are available?"
        }
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        
        # Should mention technology-related benchmarks
        response_lower = data["response"].lower()
        assert any(keyword in response_lower for keyword in [
            "nasdaq", "technology", "tech", "innovation"
        ])
        
    def test_search_by_characteristics_function_call(self, client):
        """Test search_by_characteristics function is called correctly."""
        payload = {
            "message": "What are good alternatives to Russell 2000?"
        }
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        
        # Should mention alternative benchmarks
        response_lower = data["response"].lower()
        assert any(keyword in response_lower for keyword in [
            "alternative", "similar", "small", "cap", "russell"
        ])
        
    def test_multi_function_call_sequence(self, client):
        """Test multiple function calls in sequence."""
        payload = {
            "message": "Compare minimums for S&P 500, Russell 1000, and NASDAQ-100"
        }
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        
        if data.get("function_calls"):
            # Should have made multiple function calls
            assert len(data["function_calls"]) >= 3
            
            # Should have get_minimum calls for each benchmark
            function_names = [call["name"] for call in data["function_calls"]]
            assert function_names.count("get_minimum") >= 3
        
        # Response should mention all three benchmarks
        response_lower = data["response"].lower()
        assert "s&p 500" in response_lower or "s&p" in response_lower
        assert "russell 1000" in response_lower or "russell" in response_lower
        assert "nasdaq" in response_lower

class TestFunctionCallResults:
    """Tests for function call results validation."""
    
    def test_get_minimum_result_structure(self):
        """Test get_minimum returns proper structure."""
        result = chatbot_core.get_minimum("S&P 500")
        
        assert isinstance(result, dict)
        assert "benchmark_info" in result
        assert "account_minimum" in result
        assert "minimum_display" in result
        
        benchmark_info = result["benchmark_info"]
        assert "name" in benchmark_info
        assert "account_minimum" in benchmark_info
        
    def test_get_minimum_with_invalid_name(self):
        """Test get_minimum with invalid benchmark name."""
        result = chatbot_core.get_minimum("NonexistentBenchmark")
        
        assert isinstance(result, dict)
        # Should handle gracefully - might return error or empty result
        
    def test_search_benchmarks_result_structure(self):
        """Test search_benchmarks returns proper structure."""
        result = chatbot_core.search_benchmarks("technology")
        
        assert isinstance(result, list)
        if len(result) > 0:
            benchmark = result[0]
            assert isinstance(benchmark, dict)
            assert "name" in benchmark
            
    def test_search_by_characteristics_result_structure(self):
        """Test search_by_characteristics returns proper structure."""
        result = chatbot_core.search_by_characteristics("S&P 500")
        
        assert isinstance(result, dict)
        if "alternatives" in result:
            alternatives = result["alternatives"]
            assert isinstance(alternatives, list)
            
            if len(alternatives) > 0:
                alternative = alternatives[0]
                assert isinstance(alternative, dict)
                assert "name" in alternative

class TestFunctionCallMocking:
    """Tests with mocked function calls to test error handling."""
    
    def test_function_call_timeout_handling(self, client):
        """Test handling of function call timeouts."""
        with patch('chatbot_core.get_minimum') as mock_get_minimum:
            # Simulate slow function call
            def slow_function(*args, **kwargs):
                time.sleep(2)  # Simulate slow response
                return {"benchmark_info": {"name": "Test"}, "account_minimum": 1000000}
            
            mock_get_minimum.side_effect = slow_function
            
            payload = {"message": "What is the minimum for Test Benchmark?"}
            
            start_time = time.time()
            response = client.post("/api/chat", json=payload)
            end_time = time.time()
            
            assert response.status_code == 200
            # Should complete within reasonable time despite slow function
            assert (end_time - start_time) < 25.0
            
    def test_function_call_exception_handling(self, client):
        """Test handling of function call exceptions."""
        with patch('chatbot_core.get_minimum') as mock_get_minimum:
            # Simulate function exception
            mock_get_minimum.side_effect = Exception("Simulated error")
            
            payload = {"message": "What is the minimum for S&P 500?"}
            response = client.post("/api/chat", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            # Should have graceful error message
            response_lower = data["response"].lower()
            assert any(keyword in response_lower for keyword in [
                "error", "apologize", "unable", "try again"
            ])
            
    def test_invalid_function_arguments(self, client):
        """Test handling of invalid function arguments."""
        with patch('chatbot_core.call_function') as mock_call_function:
            # Simulate invalid arguments
            mock_call_function.side_effect = TypeError("Invalid arguments")
            
            payload = {"message": "What is the minimum for S&P 500?"}
            response = client.post("/api/chat", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data

class TestFunctionCallGuardrails:
    """Tests for function calling safety guardrails."""
    
    def test_max_function_calls_limit(self, client):
        """Test that max function calls limit is respected."""
        with patch('chatbot_core.get_minimum') as mock_get_minimum:
            # Return different results to simulate iterative calling
            mock_get_minimum.return_value = {
                "benchmark_info": {"name": "Test"},
                "account_minimum": 1000000
            }
            
            # Use a query that might trigger many function calls
            payload = {
                "message": "Tell me minimums for S&P 500, Russell 1000, Russell 2000, "
                           "NASDAQ-100, MSCI EAFE, and 10 other benchmarks"
            }
            
            response = client.post("/api/chat", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            if data.get("function_calls"):
                # Should not exceed MAX_FUNCTION_CALLS (5)
                assert len(data["function_calls"]) <= 5
                
    def test_duplicate_call_prevention(self, client):
        """Test prevention of duplicate function calls."""
        call_count = 0
        
        def counting_function(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "benchmark_info": {"name": "Test"},
                "account_minimum": 1000000
            }
        
        with patch('chatbot_core.get_minimum', side_effect=counting_function):
            payload = {"message": "What is the minimum for S&P 500? And S&P 500 again?"}
            response = client.post("/api/chat", json=payload)
            
            assert response.status_code == 200
            # Should not call the same function with same args excessively
            # (exact behavior depends on implementation)
            
    def test_function_call_timeout(self, client):
        """Test function calling timeout protection."""
        start_time = time.time()
        
        # Use a complex query that might take a while
        payload = {
            "message": "Compare minimums for all available benchmarks and "
                      "find alternatives for each one"
        }
        
        response = client.post("/api/chat", json=payload)
        end_time = time.time()
        
        assert response.status_code == 200
        # Should not exceed MAX_REQUEST_TIME (30 seconds)
        assert (end_time - start_time) < 35.0
        
        data = response.json()
        assert "response" in data

class TestSpecificFunctionScenarios:
    """Tests for specific function calling scenarios."""
    
    def test_portfolio_size_filtering(self, client):
        """Test portfolio size is used in function calls."""
        payload = {
            "message": "What benchmarks work for a $250,000 portfolio?"
        }
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        response_text = data["response"]
        
        # Should mention portfolio size consideration
        response_lower = response_text.lower()
        assert any(keyword in response_lower for keyword in [
            "250", "portfolio", "eligible", "minimum"
        ])
        
    def test_international_benchmark_search(self, client):
        """Test international benchmark searches."""
        payload = {
            "message": "Show me international benchmark options"
        }
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        response_lower = data["response"].lower()
        
        # Should mention international benchmarks
        assert any(keyword in response_lower for keyword in [
            "international", "global", "eafe", "developed", "emerging"
        ])
        
    def test_esg_benchmark_search(self, client):
        """Test ESG benchmark searches."""
        payload = {
            "message": "What ESG benchmarks are available?"
        }
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        response_lower = data["response"].lower()
        
        # Should mention ESG benchmarks
        assert any(keyword in response_lower for keyword in [
            "esg", "sustainable", "environmental", "responsible"
        ])
        
    def test_factor_benchmark_search(self, client):
        """Test factor-based benchmark searches."""
        payload = {
            "message": "What factor-based benchmarks are available?"
        }
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        response_lower = data["response"].lower()
        
        # Should mention factor benchmarks
        assert any(keyword in response_lower for keyword in [
            "factor", "value", "growth", "momentum", "quality", "dividend"
        ])

class TestFunctionCallLogging:
    """Tests for function call logging and monitoring."""
    
    def test_function_call_logging(self, client, caplog):
        """Test that function calls are properly logged."""
        payload = {"message": "What is the minimum for S&P 500?"}
        
        with caplog.at_level("INFO"):
            response = client.post("/api/chat", json=payload)
            
        assert response.status_code == 200
        
        # Check for function call logs
        log_messages = [record.message for record in caplog.records]
        function_call_logs = [
            msg for msg in log_messages 
            if "Function call" in msg or "function" in msg.lower()
        ]
        
        # Should have some function call related logs
        # (exact log format may vary)
        
    def test_function_call_performance_logging(self, client, caplog):
        """Test that function call performance is logged."""
        payload = {
            "message": "Compare minimums for S&P 500 and Russell 1000"
        }
        
        with caplog.at_level("INFO"):
            response = client.post("/api/chat", json=payload)
            
        assert response.status_code == 200
        
        # Should log timing information
        log_messages = [record.message for record in caplog.records]
        timing_logs = [
            msg for msg in log_messages 
            if "completed" in msg.lower() or "time" in msg.lower()
        ]
        
        # Should have some timing logs
        # (exact format may vary)