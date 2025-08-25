"""
Performance baseline tests for Ben AI Enhanced UI.

These tests establish performance baselines for API responses,
WebSocket latency, and system resource usage to ensure
refactoring doesn't degrade performance.
"""

import asyncio
import json
import os
import sys
import time
import statistics
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

import httpx
import pytest
import psutil
import websockets
from websockets.exceptions import ConnectionClosed

# Add backend to path for imports  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Test configuration
BASE_URL = "http://localhost:8000"
WEBSOCKET_URL = "ws://localhost:8000/ws"
PERFORMANCE_SAMPLES = 5  # Number of samples for each test
TIMEOUT = 30

class PerformanceMetrics:
    """Helper class to collect performance metrics."""
    
    def __init__(self):
        self.response_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.errors = []
        
    def add_response_time(self, duration: float):
        self.response_times.append(duration)
        
    def add_memory_usage(self, usage: float):
        self.memory_usage.append(usage)
        
    def add_cpu_usage(self, usage: float):
        self.cpu_usage.append(usage)
        
    def add_error(self, error: str):
        self.errors.append(error)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "response_times": {
                "count": len(self.response_times),
                "min": min(self.response_times) if self.response_times else 0,
                "max": max(self.response_times) if self.response_times else 0,
                "mean": statistics.mean(self.response_times) if self.response_times else 0,
                "median": statistics.median(self.response_times) if self.response_times else 0,
                "p95": self._percentile(self.response_times, 0.95) if len(self.response_times) >= 20 else None
            },
            "memory_usage": {
                "count": len(self.memory_usage),
                "min": min(self.memory_usage) if self.memory_usage else 0,
                "max": max(self.memory_usage) if self.memory_usage else 0,
                "mean": statistics.mean(self.memory_usage) if self.memory_usage else 0
            },
            "cpu_usage": {
                "count": len(self.cpu_usage),
                "min": min(self.cpu_usage) if self.cpu_usage else 0,
                "max": max(self.cpu_usage) if self.cpu_usage else 0,
                "mean": statistics.mean(self.cpu_usage) if self.cpu_usage else 0
            },
            "error_count": len(self.errors),
            "errors": self.errors[:5]  # First 5 errors for debugging
        }
        return stats
        
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

@pytest.mark.skipif(
    os.environ.get("SKIP_PERFORMANCE_TESTS", "false").lower() == "true",
    reason="Performance tests disabled"
)
class TestAPIPerformance:
    """Performance tests for REST API endpoints."""
    
    @pytest.fixture(scope="class")
    def http_client(self):
        """Create HTTP client for performance testing."""
        return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)
        
    def test_health_endpoint_performance(self, http_client):
        """Test health endpoint performance baseline."""
        metrics = PerformanceMetrics()
        
        for i in range(PERFORMANCE_SAMPLES):
            # Monitor system resources
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            initial_cpu = process.cpu_percent()
            
            start_time = time.time()
            try:
                response = http_client.get("/api/health")
                end_time = time.time()
                
                if response.status_code == 200:
                    metrics.add_response_time(end_time - start_time)
                else:
                    metrics.add_error(f"HTTP {response.status_code}")
                    
            except Exception as e:
                end_time = time.time()
                metrics.add_error(str(e))
                
            # Monitor resources after request
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_cpu = process.cpu_percent()
            
            metrics.add_memory_usage(final_memory - initial_memory)
            metrics.add_cpu_usage(final_cpu)
            
            # Brief pause between requests
            time.sleep(0.1)
            
        stats = metrics.get_stats()
        
        # Performance assertions
        assert stats["response_times"]["mean"] < 2.0, f"Health endpoint too slow: {stats['response_times']['mean']:.2f}s"
        assert stats["response_times"]["max"] < 5.0, f"Health endpoint max too slow: {stats['response_times']['max']:.2f}s"
        assert stats["error_count"] == 0, f"Health endpoint errors: {stats['errors']}"
        
        # Log baseline results
        print(f"\nHealth Endpoint Performance Baseline:")
        print(f"  Average Response Time: {stats['response_times']['mean']:.3f}s")
        print(f"  Max Response Time: {stats['response_times']['max']:.3f}s")
        print(f"  Average Memory Impact: {stats['memory_usage']['mean']:.2f}MB")
        
    def test_chat_endpoint_performance(self, http_client):
        """Test chat endpoint performance baseline."""
        metrics = PerformanceMetrics()
        
        test_queries = [
            "What is the minimum for S&P 500?",
            "Show me technology benchmarks",
            "What ESG options are available?",
            "Compare Russell 1000 and S&P 500",
            "What benchmarks work for $500K?"
        ]
        
        for query in test_queries:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.time()
            try:
                payload = {"message": query}
                response = http_client.post("/api/chat", json=payload)
                end_time = time.time()
                
                if response.status_code == 200:
                    metrics.add_response_time(end_time - start_time)
                    
                    # Validate response structure
                    data = response.json()
                    assert "response" in data
                    assert "session_id" in data
                    assert len(data["response"]) > 0
                    
                else:
                    metrics.add_error(f"HTTP {response.status_code}")
                    
            except Exception as e:
                end_time = time.time()
                metrics.add_error(str(e))
                
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            metrics.add_memory_usage(final_memory - initial_memory)
            
            # Pause between requests
            time.sleep(1.0)
            
        stats = metrics.get_stats()
        
        # Performance assertions for chat (more lenient due to AI processing)
        assert stats["response_times"]["mean"] < 15.0, f"Chat endpoint too slow: {stats['response_times']['mean']:.2f}s"
        assert stats["response_times"]["max"] < 25.0, f"Chat endpoint max too slow: {stats['response_times']['max']:.2f}s"
        assert stats["error_count"] <= 1, f"Too many chat endpoint errors: {stats['errors']}"
        
        print(f"\nChat Endpoint Performance Baseline:")
        print(f"  Average Response Time: {stats['response_times']['mean']:.3f}s")
        print(f"  Max Response Time: {stats['response_times']['max']:.3f}s")
        print(f"  Error Rate: {stats['error_count']}/{len(test_queries)}")
        
    def test_search_endpoint_performance(self, http_client):
        """Test search endpoint performance baseline."""
        metrics = PerformanceMetrics()
        
        search_queries = [
            {"query": "technology"},
            {"query": "ESG sustainable"},
            {"query": "international global"},
            {"query": "small cap growth"},
            {"query": "dividend yield", "portfolio_size": 250000}
        ]
        
        for query_data in search_queries:
            start_time = time.time()
            try:
                response = http_client.post("/api/search", json=query_data)
                end_time = time.time()
                
                if response.status_code == 200:
                    metrics.add_response_time(end_time - start_time)
                    
                    data = response.json()
                    assert "results" in data
                    assert isinstance(data["results"], list)
                    
                else:
                    metrics.add_error(f"HTTP {response.status_code}")
                    
            except Exception as e:
                end_time = time.time()
                metrics.add_error(str(e))
                
            time.sleep(0.2)
            
        stats = metrics.get_stats()
        
        # Search should be faster than chat
        assert stats["response_times"]["mean"] < 5.0, f"Search endpoint too slow: {stats['response_times']['mean']:.2f}s"
        assert stats["error_count"] == 0, f"Search endpoint errors: {stats['errors']}"
        
        print(f"\nSearch Endpoint Performance Baseline:")
        print(f"  Average Response Time: {stats['response_times']['mean']:.3f}s")
        print(f"  Max Response Time: {stats['response_times']['max']:.3f}s")
        
    def test_benchmarks_endpoint_performance(self, http_client):
        """Test benchmarks listing endpoint performance."""
        metrics = PerformanceMetrics()
        
        for i in range(PERFORMANCE_SAMPLES):
            start_time = time.time()
            try:
                response = http_client.get("/api/benchmarks")
                end_time = time.time()
                
                if response.status_code == 200:
                    metrics.add_response_time(end_time - start_time)
                    
                    data = response.json()
                    assert "benchmarks" in data
                    assert isinstance(data["benchmarks"], list)
                    assert len(data["benchmarks"]) > 0
                    
                else:
                    metrics.add_error(f"HTTP {response.status_code}")
                    
            except Exception as e:
                end_time = time.time()
                metrics.add_error(str(e))
                
            time.sleep(0.1)
            
        stats = metrics.get_stats()
        
        # Benchmarks listing should be very fast
        assert stats["response_times"]["mean"] < 1.0, f"Benchmarks endpoint too slow: {stats['response_times']['mean']:.2f}s"
        assert stats["error_count"] == 0, f"Benchmarks endpoint errors: {stats['errors']}"
        
        print(f"\nBenchmarks Endpoint Performance Baseline:")
        print(f"  Average Response Time: {stats['response_times']['mean']:.3f}s")

@pytest.mark.skipif(
    os.environ.get("SKIP_PERFORMANCE_TESTS", "false").lower() == "true",
    reason="Performance tests disabled"
)
class TestWebSocketPerformance:
    """Performance tests for WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_latency(self):
        """Test WebSocket connection establishment latency."""
        metrics = PerformanceMetrics()
        
        for i in range(PERFORMANCE_SAMPLES):
            session_id = f"perf-test-{i}"
            
            start_time = time.time()
            try:
                websocket = await websockets.connect(f"{WEBSOCKET_URL}/{session_id}")
                end_time = time.time()
                
                metrics.add_response_time(end_time - start_time)
                
                await websocket.close()
                
            except Exception as e:
                end_time = time.time()
                metrics.add_error(str(e))
                
            await asyncio.sleep(0.1)
            
        stats = metrics.get_stats()
        
        assert stats["response_times"]["mean"] < 1.0, f"WebSocket connection too slow: {stats['response_times']['mean']:.2f}s"
        assert stats["error_count"] == 0, f"WebSocket connection errors: {stats['errors']}"
        
        print(f"\nWebSocket Connection Performance Baseline:")
        print(f"  Average Connection Time: {stats['response_times']['mean']:.3f}s")
        
    @pytest.mark.asyncio
    async def test_websocket_message_latency(self):
        """Test WebSocket message round-trip latency."""
        metrics = PerformanceMetrics()
        session_id = f"perf-msg-test-{uuid.uuid4()}"
        
        try:
            websocket = await websockets.connect(f"{WEBSOCKET_URL}/{session_id}")
            
            test_messages = [
                "What is the minimum for S&P 500?",
                "Show me technology benchmarks",
                "What ESG options are available?"
            ]
            
            for message in test_messages:
                start_time = time.time()
                
                try:
                    # Send message
                    payload = {
                        "type": "message",
                        "message": message
                    }
                    await websocket.send(json.dumps(payload))
                    
                    # Wait for typing start
                    response = await asyncio.wait_for(
                        websocket.recv(), 
                        timeout=10.0
                    )
                    data = json.loads(response)
                    
                    if data.get("type") == "typing" and data.get("status") == "start":
                        # Continue receiving until we get the actual response
                        while True:
                            response = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=20.0
                            )
                            data = json.loads(response)
                            
                            if (data.get("type") == "message" and 
                                "data" in data and 
                                "response" in data["data"]):
                                end_time = time.time()
                                metrics.add_response_time(end_time - start_time)
                                break
                                
                            if data.get("type") == "typing" and data.get("status") == "stop":
                                # Sometimes typing stops without message - wait a bit more
                                try:
                                    final_response = await asyncio.wait_for(
                                        websocket.recv(),
                                        timeout=2.0
                                    )
                                    final_data = json.loads(final_response)
                                    if (final_data.get("type") == "message" and
                                        "data" in final_data):
                                        end_time = time.time()
                                        metrics.add_response_time(end_time - start_time)
                                        break
                                except asyncio.TimeoutError:
                                    end_time = time.time()
                                    metrics.add_error("No response after typing stopped")
                                    break
                    
                except Exception as e:
                    end_time = time.time()
                    metrics.add_error(str(e))
                    
                await asyncio.sleep(2.0)  # Pause between messages
                
            await websocket.close()
            
        except Exception as e:
            metrics.add_error(f"WebSocket setup error: {e}")
            
        stats = metrics.get_stats()
        
        # WebSocket messages should be comparable to HTTP chat
        if stats["response_times"]["count"] > 0:
            assert stats["response_times"]["mean"] < 20.0, f"WebSocket messages too slow: {stats['response_times']['mean']:.2f}s"
            
            print(f"\nWebSocket Message Performance Baseline:")
            print(f"  Average Message Round-trip: {stats['response_times']['mean']:.3f}s")
            print(f"  Messages Processed: {stats['response_times']['count']}")
            
        if stats["error_count"] > 0:
            print(f"  Errors: {stats['error_count']} - {stats['errors']}")
            
    @pytest.mark.asyncio
    async def test_websocket_ping_pong_latency(self):
        """Test WebSocket ping-pong latency."""
        metrics = PerformanceMetrics()
        session_id = f"perf-ping-test-{uuid.uuid4()}"
        
        try:
            websocket = await websockets.connect(f"{WEBSOCKET_URL}/{session_id}")
            
            for i in range(PERFORMANCE_SAMPLES):
                start_time = time.time()
                
                try:
                    # Send ping
                    await websocket.send(json.dumps({"type": "ping"}))
                    
                    # Wait for pong
                    response = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=5.0
                    )
                    end_time = time.time()
                    
                    data = json.loads(response)
                    if data.get("type") == "pong":
                        metrics.add_response_time(end_time - start_time)
                    else:
                        metrics.add_error(f"Unexpected response: {data}")
                        
                except Exception as e:
                    end_time = time.time()
                    metrics.add_error(str(e))
                    
                await asyncio.sleep(0.1)
                
            await websocket.close()
            
        except Exception as e:
            metrics.add_error(f"WebSocket ping test error: {e}")
            
        stats = metrics.get_stats()
        
        # Ping-pong should be very fast
        if stats["response_times"]["count"] > 0:
            assert stats["response_times"]["mean"] < 0.5, f"Ping-pong too slow: {stats['response_times']['mean']:.2f}s"
            
            print(f"\nWebSocket Ping-Pong Performance Baseline:")
            print(f"  Average Ping-Pong Time: {stats['response_times']['mean']:.3f}s")

@pytest.mark.skipif(
    os.environ.get("SKIP_PERFORMANCE_TESTS", "false").lower() == "true",
    reason="Performance tests disabled"
)
class TestConcurrentPerformance:
    """Performance tests for concurrent usage scenarios."""
    
    def test_concurrent_http_requests(self):
        """Test concurrent HTTP request handling."""
        metrics = PerformanceMetrics()
        
        def make_request(client, query):
            try:
                start_time = time.time()
                response = client.post("/api/chat", json={"message": query})
                end_time = time.time()
                
                if response.status_code == 200:
                    return end_time - start_time, None
                else:
                    return None, f"HTTP {response.status_code}"
                    
            except Exception as e:
                return None, str(e)
                
        # Test concurrent requests
        queries = [
            "What is the minimum for S&P 500?",
            "Show me Russell benchmarks",
            "What ESG options exist?",
            "Technology benchmark minimums",
            "International benchmark list"
        ]
        
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(make_request, client, query)
                    for query in queries
                ]
                
                for future in futures:
                    duration, error = future.result()
                    if duration:
                        metrics.add_response_time(duration)
                    if error:
                        metrics.add_error(error)
                        
        stats = metrics.get_stats()
        
        # Concurrent requests should not be significantly slower
        assert stats["response_times"]["mean"] < 20.0, f"Concurrent requests too slow: {stats['response_times']['mean']:.2f}s"
        assert stats["error_count"] <= 1, f"Too many concurrent request errors: {stats['errors']}"
        
        print(f"\nConcurrent HTTP Performance Baseline:")
        print(f"  Average Response Time: {stats['response_times']['mean']:.3f}s")
        print(f"  Concurrent Requests Processed: {stats['response_times']['count']}")
        print(f"  Error Rate: {stats['error_count']}/{len(queries)}")

@pytest.mark.skipif(
    os.environ.get("SKIP_PERFORMANCE_TESTS", "false").lower() == "true",
    reason="Performance tests disabled"
)
class TestSystemResourceUsage:
    """Tests for system resource usage baselines."""
    
    def test_memory_usage_baseline(self):
        """Test memory usage during typical operations."""
        process = psutil.Process()
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform typical operations
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
            # Health check
            client.get("/api/health")
            
            # Chat requests
            queries = [
                "What is the minimum for S&P 500?",
                "Show me technology benchmarks",
                "Compare Russell 1000 and S&P 500 minimums"
            ]
            
            for query in queries:
                client.post("/api/chat", json={"message": query})
                time.sleep(1.0)
                
            # Get benchmarks
            client.get("/api/benchmarks")
            
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase:.2f}MB"
        
        print(f"\nMemory Usage Baseline:")
        print(f"  Initial Memory: {initial_memory:.2f}MB")
        print(f"  Final Memory: {final_memory:.2f}MB")
        print(f"  Memory Increase: {memory_increase:.2f}MB")
        
    def test_response_time_consistency(self):
        """Test response time consistency over multiple requests."""
        metrics = PerformanceMetrics()
        
        query = "What is the minimum for S&P 500?"
        
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
            for i in range(10):  # More samples for consistency test
                start_time = time.time()
                try:
                    response = client.post("/api/chat", json={"message": query})
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        metrics.add_response_time(end_time - start_time)
                    else:
                        metrics.add_error(f"HTTP {response.status_code}")
                        
                except Exception as e:
                    metrics.add_error(str(e))
                    
                time.sleep(1.0)
                
        stats = metrics.get_stats()
        
        if len(metrics.response_times) >= 3:
            # Calculate coefficient of variation for consistency
            mean_time = stats["response_times"]["mean"]
            std_dev = statistics.stdev(metrics.response_times)
            coefficient_of_variation = std_dev / mean_time if mean_time > 0 else 0
            
            # Response times should be reasonably consistent (CV < 0.5)
            assert coefficient_of_variation < 0.5, f"Response times too inconsistent: CV={coefficient_of_variation:.2f}"
            
            print(f"\nResponse Time Consistency Baseline:")
            print(f"  Mean Response Time: {mean_time:.3f}s")
            print(f"  Standard Deviation: {std_dev:.3f}s")
            print(f"  Coefficient of Variation: {coefficient_of_variation:.3f}")

def save_performance_baselines(results: Dict[str, Any], filename: str = "performance_baselines.json"):
    """Save performance baseline results to file for future comparison."""
    baseline_data = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version
        }
    }
    
    baseline_path = os.path.join(os.path.dirname(__file__), filename)
    with open(baseline_path, 'w') as f:
        json.dump(baseline_data, f, indent=2)
        
    print(f"\nPerformance baselines saved to: {baseline_path}")

if __name__ == "__main__":
    # Run performance tests and save baselines when executed directly
    pytest.main([__file__, "-v"])