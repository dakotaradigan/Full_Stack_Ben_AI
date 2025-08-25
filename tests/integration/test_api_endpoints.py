"""
Integration tests for API endpoints.

These tests validate the REST API endpoints are working correctly
with actual backend services running.
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, Any

import httpx
import pytest
from fastapi.testclient import TestClient

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from app import app

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

@pytest.fixture(scope="module")
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)

@pytest.fixture(scope="module") 
def http_client():
    """Create HTTP client for live server testing."""
    return httpx.Client(base_url=BASE_URL, timeout=TEST_TIMEOUT)

class TestHealthEndpoint:
    """Tests for /api/health endpoint."""
    
    def test_health_check_success(self, client):
        """Test health endpoint returns success status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]
        assert "backend" in data
        
    def test_health_check_structure(self, client):
        """Test health endpoint response structure."""
        response = client.get("/api/health")
        data = response.json()
        
        # Required fields
        required_fields = ["status", "backend"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
            
    @pytest.mark.timeout(10)
    def test_health_check_performance(self, client):
        """Test health endpoint responds within acceptable time."""
        start_time = time.time()
        response = client.get("/api/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 5.0, "Health check took too long"

class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "active"

class TestChatEndpoint:
    """Tests for /api/chat endpoint."""
    
    def test_chat_simple_message(self, client):
        """Test chat endpoint with simple message."""
        payload = {
            "message": "Hello, what is the minimum for S&P 500?"
        }
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        assert "session_id" in data
        assert "timestamp" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0
        
    def test_chat_with_session_id(self, client):
        """Test chat endpoint maintains session."""
        session_id = "test-session-123"
        payload = {
            "message": "What is the minimum for Russell 1000?",
            "session_id": session_id
        }
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["session_id"] == session_id
        
    def test_chat_invalid_input(self, client):
        """Test chat endpoint handles invalid input."""
        payload = {}  # Missing message
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 422  # Validation error
        
    def test_chat_empty_message(self, client):
        """Test chat endpoint handles empty message."""
        payload = {"message": ""}
        
        response = client.post("/api/chat", json=payload)
        # Should still return 200 but with appropriate response
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        
    def test_chat_long_message(self, client):
        """Test chat endpoint handles long messages."""
        long_message = "Tell me about benchmarks. " * 100  # ~3000 chars
        payload = {"message": long_message}
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        
    def test_chat_function_calling(self, client):
        """Test chat endpoint triggers function calls."""
        payload = {
            "message": "What is the minimum investment for S&P 500?"
        }
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        # Should mention specific minimum amount
        response_lower = data["response"].lower()
        assert any(keyword in response_lower for keyword in ["million", "minimum", "$", "investment"])
        
    def test_chat_multi_benchmark_comparison(self, client):
        """Test chat endpoint handles multi-benchmark queries."""
        payload = {
            "message": "Compare minimums for S&P 500, Russell 1000, and NASDAQ-100"
        }
        
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "response" in data
        # Should mention all three benchmarks
        response_lower = data["response"].lower()
        assert "s&p 500" in response_lower or "s&p" in response_lower
        assert "russell" in response_lower
        assert "nasdaq" in response_lower
        
    @pytest.mark.timeout(30)
    def test_chat_performance(self, client):
        """Test chat endpoint performance."""
        payload = {"message": "What is the minimum for Russell 2000?"}
        
        start_time = time.time()
        response = client.post("/api/chat", json=payload)
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 20.0, "Chat response took too long"

class TestSuggestionsEndpoint:
    """Tests for /api/suggestions endpoint."""
    
    def test_suggestions_structure(self, client):
        """Test suggestions endpoint returns proper structure."""
        response = client.get("/api/suggestions")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check main categories exist
        expected_categories = ["getting_started", "use_cases", "benchmark_search"]
        for category in expected_categories:
            assert category in data
            assert isinstance(data[category], list)
            
        # Check suggestion structure
        for category in expected_categories:
            for suggestion in data[category]:
                assert "title" in suggestion
                assert "query" in suggestion
                assert isinstance(suggestion["title"], str)
                assert isinstance(suggestion["query"], str)
                assert len(suggestion["title"]) > 0
                assert len(suggestion["query"]) > 0

class TestSearchEndpoint:
    """Tests for /api/search endpoint."""
    
    def test_search_basic_query(self, client):
        """Test search endpoint with basic query."""
        payload = {
            "query": "technology benchmarks"
        }
        
        response = client.post("/api/search", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        
    def test_search_with_portfolio_size(self, client):
        """Test search endpoint with portfolio size filter."""
        payload = {
            "query": "ESG benchmarks",
            "portfolio_size": 250000
        }
        
        response = client.post("/api/search", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        
    def test_search_with_filters(self, client):
        """Test search endpoint with additional filters."""
        payload = {
            "query": "international benchmarks", 
            "filters": {
                "region": ["International"]
            }
        }
        
        response = client.post("/api/search", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        
    def test_search_invalid_input(self, client):
        """Test search endpoint handles invalid input."""
        payload = {}  # Missing query
        
        response = client.post("/api/search", json=payload)
        assert response.status_code == 422  # Validation error

class TestBenchmarksEndpoint:
    """Tests for /api/benchmarks endpoint."""
    
    def test_list_benchmarks(self, client):
        """Test benchmarks endpoint lists benchmarks."""
        response = client.get("/api/benchmarks")
        assert response.status_code == 200
        
        data = response.json()
        assert "benchmarks" in data
        assert isinstance(data["benchmarks"], list)
        assert len(data["benchmarks"]) > 0
        
        # Check benchmark structure
        for benchmark in data["benchmarks"]:
            assert "name" in benchmark
            assert "account_minimum" in benchmark
            assert "market_cap" in benchmark
            assert isinstance(benchmark["name"], str)
            assert len(benchmark["name"]) > 0

class TestErrorHandling:
    """Tests for error handling across endpoints."""
    
    def test_invalid_endpoint(self, client):
        """Test accessing non-existent endpoint."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        
    def test_method_not_allowed(self, client):
        """Test invalid HTTP method."""
        response = client.delete("/api/chat")  # DELETE not allowed
        assert response.status_code == 405
        
    def test_malformed_json(self, client):
        """Test handling of malformed JSON."""
        response = client.post(
            "/api/chat", 
            data="invalid json content",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422

class TestLiveServerIntegration:
    """Tests against live server (requires server running on localhost:8000)."""
    
    @pytest.mark.skipif(
        os.environ.get("SKIP_LIVE_TESTS", "false").lower() == "true",
        reason="Live server tests disabled"
    )
    def test_live_health_check(self, http_client):
        """Test health check against live server."""
        try:
            response = http_client.get("/api/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            
        except httpx.ConnectError:
            pytest.skip("Live server not available at localhost:8000")
            
    @pytest.mark.skipif(
        os.environ.get("SKIP_LIVE_TESTS", "false").lower() == "true",
        reason="Live server tests disabled"
    )
    def test_live_chat_integration(self, http_client):
        """Test chat integration against live server."""
        try:
            payload = {"message": "What is the minimum for S&P 500?"}
            response = http_client.post("/api/chat", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert "response" in data
            assert len(data["response"]) > 0
            
        except httpx.ConnectError:
            pytest.skip("Live server not available at localhost:8000")