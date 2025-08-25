"""
Integration tests for WebSocket functionality.

These tests validate WebSocket connections, message handling,
and real-time chat functionality.
"""

import asyncio
import json
import os
import sys
import time
import uuid
from typing import List, Dict, Any

import pytest
import websockets
from websockets.exceptions import ConnectionClosed

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Test configuration
WEBSOCKET_URL = "ws://localhost:8000/ws"
TEST_TIMEOUT = 30

class WebSocketTestClient:
    """Helper class for WebSocket testing."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.url = f"{WEBSOCKET_URL}/{session_id}"
        self.websocket = None
        self.received_messages = []
        
    async def connect(self):
        """Connect to WebSocket."""
        self.websocket = await websockets.connect(self.url)
        
    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
            
    async def send_message(self, message: str):
        """Send a chat message."""
        payload = {
            "type": "message",
            "message": message
        }
        await self.websocket.send(json.dumps(payload))
        
    async def send_ping(self):
        """Send a ping message."""
        payload = {"type": "ping"}
        await self.websocket.send(json.dumps(payload))
        
    async def receive_message(self, timeout: float = 10.0):
        """Receive a message with timeout."""
        try:
            message = await asyncio.wait_for(
                self.websocket.recv(), 
                timeout=timeout
            )
            parsed_message = json.loads(message)
            self.received_messages.append(parsed_message)
            return parsed_message
        except asyncio.TimeoutError:
            raise TimeoutError(f"No message received within {timeout} seconds")
            
    async def receive_until_response(self, timeout: float = 30.0):
        """Receive messages until we get a chat response."""
        start_time = time.time()
        messages = []
        
        while time.time() - start_time < timeout:
            try:
                message = await self.receive_message(timeout=5.0)
                messages.append(message)
                
                # Check for chat response
                if (message.get("type") == "message" and 
                    "data" in message and 
                    "response" in message["data"]):
                    return messages
                    
                # Also return if typing stops (end of conversation)
                if (message.get("type") == "typing" and 
                    message.get("status") == "stop"):
                    # Wait a bit more for the actual response
                    try:
                        final_message = await self.receive_message(timeout=2.0)
                        messages.append(final_message)
                        if (final_message.get("type") == "message" and 
                            "data" in final_message):
                            return messages
                    except TimeoutError:
                        pass
                    return messages
                    
            except TimeoutError:
                break
                
        raise TimeoutError(f"No chat response received within {timeout} seconds")

class TestWebSocketConnection:
    """Tests for WebSocket connection management."""
    
    @pytest.mark.asyncio
    async def test_websocket_connect_disconnect(self):
        """Test basic WebSocket connection and disconnection."""
        session_id = str(uuid.uuid4())
        client = WebSocketTestClient(session_id)
        
        # Test connection
        await client.connect()
        assert client.websocket is not None
        
        # Test disconnection
        await client.disconnect()
        
    @pytest.mark.asyncio
    async def test_multiple_sessions(self):
        """Test multiple WebSocket sessions simultaneously."""
        session1 = str(uuid.uuid4())
        session2 = str(uuid.uuid4())
        
        client1 = WebSocketTestClient(session1)
        client2 = WebSocketTestClient(session2)
        
        try:
            # Connect both clients
            await client1.connect()
            await client2.connect()
            
            # Both should be connected
            assert client1.websocket is not None
            assert client2.websocket is not None
            
        finally:
            await client1.disconnect()
            await client2.disconnect()
            
    @pytest.mark.asyncio
    async def test_invalid_session_id(self):
        """Test connection with various session ID formats."""
        # Test with empty session ID - should still work
        client = WebSocketTestClient("")
        await client.connect()
        await client.disconnect()
        
        # Test with special characters
        client = WebSocketTestClient("session-with-dashes")
        await client.connect()
        await client.disconnect()

class TestWebSocketMessaging:
    """Tests for WebSocket message handling."""
    
    @pytest.mark.asyncio
    async def test_ping_pong(self):
        """Test ping-pong mechanism."""
        session_id = str(uuid.uuid4())
        client = WebSocketTestClient(session_id)
        
        try:
            await client.connect()
            
            # Send ping
            await client.send_ping()
            
            # Should receive pong
            response = await client.receive_message(timeout=5.0)
            assert response["type"] == "pong"
            
        finally:
            await client.disconnect()
            
    @pytest.mark.asyncio
    async def test_simple_chat_message(self):
        """Test sending and receiving a simple chat message."""
        session_id = str(uuid.uuid4())
        client = WebSocketTestClient(session_id)
        
        try:
            await client.connect()
            
            # Send message
            await client.send_message("What is the minimum for S&P 500?")
            
            # Receive response sequence
            messages = await client.receive_until_response()
            
            # Should get typing indicators and response
            message_types = [msg.get("type") for msg in messages]
            assert "typing" in message_types
            assert "message" in message_types
            
            # Find the actual response
            response_messages = [
                msg for msg in messages 
                if msg.get("type") == "message" and "data" in msg
            ]
            assert len(response_messages) >= 1
            
            response_data = response_messages[-1]["data"]
            assert "response" in response_data
            assert len(response_data["response"]) > 0
            assert "session_id" in response_data
            assert response_data["session_id"] == session_id
            
        finally:
            await client.disconnect()
            
    @pytest.mark.asyncio
    async def test_typing_indicators(self):
        """Test typing indicator sequence."""
        session_id = str(uuid.uuid4())
        client = WebSocketTestClient(session_id)
        
        try:
            await client.connect()
            
            # Send message
            await client.send_message("Hello")
            
            # First message should be typing start
            first_message = await client.receive_message()
            assert first_message["type"] == "typing"
            assert first_message["status"] == "start"
            
            # Eventually should get typing stop
            messages = await client.receive_until_response()
            typing_messages = [
                msg for msg in messages 
                if msg.get("type") == "typing"
            ]
            
            # Should have both start and stop
            statuses = [msg.get("status") for msg in typing_messages]
            assert "start" in statuses
            assert "stop" in statuses
            
        finally:
            await client.disconnect()
            
    @pytest.mark.asyncio
    async def test_session_persistence(self):
        """Test that session persists across messages."""
        session_id = str(uuid.uuid4())
        client = WebSocketTestClient(session_id)
        
        try:
            await client.connect()
            
            # Send first message
            await client.send_message("What is the minimum for Russell 1000?")
            messages1 = await client.receive_until_response()
            
            # Send second message
            await client.send_message("What about S&P 500?")
            messages2 = await client.receive_until_response()
            
            # Both should have same session_id
            response1 = [
                msg for msg in messages1 
                if msg.get("type") == "message" and "data" in msg
            ][-1]
            
            response2 = [
                msg for msg in messages2 
                if msg.get("type") == "message" and "data" in msg
            ][-1]
            
            assert response1["data"]["session_id"] == session_id
            assert response2["data"]["session_id"] == session_id
            
        finally:
            await client.disconnect()
            
    @pytest.mark.asyncio
    async def test_invalid_message_format(self):
        """Test handling of invalid message formats."""
        session_id = str(uuid.uuid4())
        client = WebSocketTestClient(session_id)
        
        try:
            await client.connect()
            
            # Send invalid JSON
            await client.websocket.send("invalid json")
            
            # Connection should remain open (no response expected)
            # Send valid message to test connection is still working
            await client.send_message("Hello")
            messages = await client.receive_until_response()
            
            # Should still work
            assert len(messages) > 0
            
        finally:
            await client.disconnect()
            
    @pytest.mark.asyncio
    async def test_empty_message(self):
        """Test handling of empty messages."""
        session_id = str(uuid.uuid4())
        client = WebSocketTestClient(session_id)
        
        try:
            await client.connect()
            
            # Send empty message
            await client.send_message("")
            
            # Should still get response
            messages = await client.receive_until_response()
            response_messages = [
                msg for msg in messages 
                if msg.get("type") == "message" and "data" in msg
            ]
            
            assert len(response_messages) >= 1
            
        finally:
            await client.disconnect()

class TestWebSocketFunctionCalling:
    """Tests for function calling through WebSocket."""
    
    @pytest.mark.asyncio
    async def test_function_calling_response(self):
        """Test that function calls work through WebSocket."""
        session_id = str(uuid.uuid4())
        client = WebSocketTestClient(session_id)
        
        try:
            await client.connect()
            
            # Ask for specific benchmark minimum
            await client.send_message("What is the minimum investment for Russell 2000?")
            
            messages = await client.receive_until_response(timeout=30.0)
            
            # Find the response message
            response_messages = [
                msg for msg in messages 
                if msg.get("type") == "message" and "data" in msg
            ]
            assert len(response_messages) >= 1
            
            response_data = response_messages[-1]["data"]
            response_text = response_data["response"].lower()
            
            # Should mention minimum amount
            assert any(keyword in response_text for keyword in [
                "million", "minimum", "$", "investment"
            ])
            
        finally:
            await client.disconnect()
            
    @pytest.mark.asyncio
    async def test_multi_benchmark_query(self):
        """Test multi-benchmark comparison through WebSocket."""
        session_id = str(uuid.uuid4())
        client = WebSocketTestClient(session_id)
        
        try:
            await client.connect()
            
            # Ask for comparison
            await client.send_message("Compare minimums for S&P 500 and Russell 1000")
            
            messages = await client.receive_until_response(timeout=30.0)
            
            # Find the response message
            response_messages = [
                msg for msg in messages 
                if msg.get("type") == "message" and "data" in msg
            ]
            assert len(response_messages) >= 1
            
            response_data = response_messages[-1]["data"]
            response_text = response_data["response"].lower()
            
            # Should mention both benchmarks
            assert "s&p" in response_text or "s&p 500" in response_text
            assert "russell" in response_text
            
        finally:
            await client.disconnect()

class TestWebSocketErrorHandling:
    """Tests for WebSocket error handling."""
    
    @pytest.mark.asyncio
    async def test_connection_recovery(self):
        """Test connection behavior after errors."""
        session_id = str(uuid.uuid4())
        client = WebSocketTestClient(session_id)
        
        try:
            await client.connect()
            
            # Send valid message first
            await client.send_message("Hello")
            messages = await client.receive_until_response()
            assert len(messages) > 0
            
            # Connection should still work
            await client.send_message("What is the minimum for S&P 500?")
            messages2 = await client.receive_until_response()
            assert len(messages2) > 0
            
        finally:
            await client.disconnect()
            
    @pytest.mark.asyncio
    async def test_large_message_handling(self):
        """Test handling of large messages."""
        session_id = str(uuid.uuid4())
        client = WebSocketTestClient(session_id)
        
        try:
            await client.connect()
            
            # Send large message
            large_message = "Tell me about benchmarks. " * 200  # ~6000 chars
            await client.send_message(large_message)
            
            # Should still get response
            messages = await client.receive_until_response(timeout=30.0)
            response_messages = [
                msg for msg in messages 
                if msg.get("type") == "message" and "data" in msg
            ]
            
            assert len(response_messages) >= 1
            
        finally:
            await client.disconnect()

class TestWebSocketPerformance:
    """Performance tests for WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_response_time(self):
        """Test WebSocket response time."""
        session_id = str(uuid.uuid4())
        client = WebSocketTestClient(session_id)
        
        try:
            await client.connect()
            
            start_time = time.time()
            await client.send_message("What is the minimum for S&P 500?")
            
            messages = await client.receive_until_response(timeout=20.0)
            end_time = time.time()
            
            response_time = end_time - start_time
            assert response_time < 15.0, f"Response took too long: {response_time}s"
            
            # Should have received messages
            assert len(messages) > 0
            
        finally:
            await client.disconnect()
            
    @pytest.mark.asyncio
    async def test_concurrent_messages(self):
        """Test handling of multiple concurrent sessions."""
        num_clients = 3
        clients = []
        
        try:
            # Create and connect multiple clients
            for i in range(num_clients):
                session_id = f"concurrent-test-{i}"
                client = WebSocketTestClient(session_id)
                await client.connect()
                clients.append(client)
            
            # Send messages simultaneously
            tasks = []
            for i, client in enumerate(clients):
                task = asyncio.create_task(
                    client.send_message(f"What is the minimum for benchmark {i}?")
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # All should get responses
            for client in clients:
                messages = await client.receive_until_response(timeout=30.0)
                assert len(messages) > 0
                
        finally:
            # Clean up all clients
            for client in clients:
                try:
                    await client.disconnect()
                except:
                    pass  # Ignore cleanup errors

@pytest.mark.skipif(
    os.environ.get("SKIP_LIVE_TESTS", "false").lower() == "true",
    reason="Live server tests disabled"
)
class TestLiveWebSocketIntegration:
    """Tests against live WebSocket server."""
    
    @pytest.mark.asyncio
    async def test_live_websocket_connection(self):
        """Test connection to live WebSocket server."""
        session_id = str(uuid.uuid4())
        
        try:
            websocket = await websockets.connect(f"{WEBSOCKET_URL}/{session_id}")
            
            # Test ping
            await websocket.send(json.dumps({"type": "ping"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            assert data["type"] == "pong"
            
            await websocket.close()
            
        except ConnectionRefusedError:
            pytest.skip("Live WebSocket server not available")
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")