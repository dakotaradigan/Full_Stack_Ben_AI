from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import asyncio
import logging
import time
from datetime import datetime
import uuid

# Import chatbot core functionality
from chatbot_core import (
    sanitize_input,
    validate_response_security,
    call_function,
    FUNCTIONS,
    SYSTEM_PROMPT,
    DISCLAIMER_TEXT,
    DISCLAIMER_FREQUENCY,
    client,
    CHAT_MODEL,
    num_tokens_from_messages,
    trim_history,
    MAX_TOKENS_PER_REQUEST,
    get_minimum,
    search_by_characteristics,
    search_benchmarks
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for iterative function calling safety
MAX_FUNCTION_CALLS = 5  # Limit to 5 benchmarks per query for cost control
MAX_REQUEST_TIME = 30   # 30 second timeout for function calling
DUPLICATE_CALL_LIMIT = 2  # Prevent infinite loops

# Initialize FastAPI app
app = FastAPI(title="Ben AI Chatbot API", version="1.0.0")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    function_calls: Optional[List[Dict]] = None

class SearchQuery(BaseModel):
    query: str
    portfolio_size: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None

# Session management
sessions = {}

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []
        self.interaction_count = 0
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.last_activity = datetime.now()
        if role == "assistant":
            self.interaction_count += 1
    
    def should_show_disclaimer(self):
        return self.interaction_count % DISCLAIMER_FREQUENCY == 0 and self.interaction_count > 0
    
    def get_trimmed_history(self):
        # Create a copy of messages to avoid modifying the original
        messages_copy = self.messages.copy()
        trim_history(messages_copy, MAX_TOKENS_PER_REQUEST - 500)
        return messages_copy

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Ben AI Chatbot API", "status": "active"}

@app.get("/api/health")
async def health_check():
    try:
        # Test backend connection
        result = get_minimum("Russell 2000")
        return {"status": "healthy", "backend": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        # Sanitize input
        sanitized_message = sanitize_input(message.message)
        
        # Get or create session
        session_id = message.session_id or str(uuid.uuid4())
        if session_id not in sessions:
            sessions[session_id] = ChatSession(session_id)
        
        session = sessions[session_id]
        
        # Add user message
        session.add_message("user", sanitized_message)
        
        # Get conversation history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(session.get_trimmed_history())
        
        # Add disclaimer if needed
        disclaimer = ""
        if session.should_show_disclaimer():
            disclaimer = f"\n\n*{DISCLAIMER_TEXT}*"
        
        # Get AI response
        try:
            completion = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                functions=FUNCTIONS,
                function_call="auto",
                temperature=0.1,
                max_tokens=min(MAX_TOKENS_PER_REQUEST, 2000)
            )
            
            response_message = completion.choices[0].message
            function_calls = []
            
            # Handle iterative function calls with safety guardrails
            final_response = None
            start_time = time.time()
            call_history = {}  # Track duplicate calls to prevent infinite loops
            current_response = response_message
            
            # Iterative function calling loop with safety guardrails
            for iteration in range(MAX_FUNCTION_CALLS):
                # GUARDRAIL 1: Timeout protection
                if time.time() - start_time > MAX_REQUEST_TIME:
                    logger.warning(f"Function calling timeout after {iteration} iterations - stopping")
                    break
                
                # GUARDRAIL 2: Check if AI wants to make a function call
                if not current_response.function_call:
                    # No more function calls requested - AI is done
                    final_response = current_response.content
                    break
                
                # Extract function call details
                function_name = current_response.function_call.name
                function_args = json.loads(current_response.function_call.arguments)
                
                # GUARDRAIL 3: Duplicate call detection (infinite loop prevention)
                call_signature = f"{function_name}:{json.dumps(function_args, sort_keys=True)}"
                call_history[call_signature] = call_history.get(call_signature, 0) + 1
                
                if call_history[call_signature] > DUPLICATE_CALL_LIMIT:
                    logger.warning(f"Duplicate function call detected after {iteration} iterations - stopping: {call_signature}")
                    final_response = current_response.content if current_response.content else "I have the information you requested."
                    break
                
                # Execute function safely
                try:
                    function_response = call_function(function_name, function_args)
                    function_calls.append({
                        "name": function_name,
                        "args": function_args,
                        "result": function_response
                    })
                    
                    logger.info(f"Function call {iteration + 1}: {function_name} executed successfully")
                    
                except Exception as e:
                    logger.error(f"Function call failed: {function_name} - {e}")
                    final_response = "I apologize, but I encountered an error while looking up that information."
                    break
                
                # Add function call and result to conversation context
                messages.append(current_response.model_dump())
                messages.append({
                    "role": "function", 
                    "name": function_name,
                    "content": json.dumps(function_response)
                })
                
                # GUARDRAIL 4: Token limit protection
                estimated_tokens = len(json.dumps(messages)) // 4  # Rough estimate
                if estimated_tokens > MAX_TOKENS_PER_REQUEST - 500:
                    logger.warning(f"Approaching token limit after {iteration + 1} function calls - stopping")
                    final_response = "I have gathered the available information for you."
                    break
                
                # Get next AI response (might request another function call)
                try:
                    next_completion = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=messages,
                        functions=FUNCTIONS,
                        function_call="auto",
                        temperature=0.1,
                        max_tokens=min(MAX_TOKENS_PER_REQUEST, 2000)
                    )
                    current_response = next_completion.choices[0].message
                    
                except Exception as e:
                    logger.error(f"OpenAI API error during iteration {iteration + 1}: {e}")
                    final_response = "I have the information you requested."
                    break
            
            # Handle case where we hit max iterations without AI finishing
            if final_response is None:
                logger.warning(f"Reached maximum function calls ({MAX_FUNCTION_CALLS}) - using last AI response")
                final_response = current_response.content if current_response.content else "I have gathered the available information for you."
            
            # Log summary of function calling session
            if function_calls:
                logger.info(f"Function calling session completed: {len(function_calls)} calls in {time.time() - start_time:.2f}s")
            
            # Ensure we have a valid response
            if final_response is None:
                final_response = "I understand your question. Let me help you with benchmark eligibility information."
            
            # Validate response security (pass function call success status)
            function_calls_successful = len(function_calls) > 0
            final_response = validate_response_security(final_response, function_calls_successful)
            
            # Add disclaimer if needed
            if disclaimer:
                final_response += disclaimer
            
            # Add assistant message to session
            session.add_message("assistant", final_response)
            
            return ChatResponse(
                response=final_response,
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                function_calls=function_calls if function_calls else None
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            error_response = "I apologize, but I encountered an error processing your request. Please try again."
            session.add_message("assistant", error_response)
            return ChatResponse(
                response=error_response,
                session_id=session_id,
                timestamp=datetime.now().isoformat()
            )
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/suggestions")
async def get_suggestions():
    """Return sample queries for the sidebar"""
    suggestions = {
        "getting_started": [
            {
                "title": "Check S&P 500 minimum",
                "query": "What's the minimum investment for S&P 500?"
            },
            {
                "title": "Find ESG benchmarks",
                "query": "Show me ESG-focused benchmark options"
            },
            {
                "title": "Small-cap alternatives",
                "query": "What are the best small-cap benchmarks?"
            }
        ],
        "portfolio_analysis": [
            {
                "title": "Check portfolio eligibility",
                "query": "Can a $250,000 portfolio use Russell 2000?"
            },
            {
                "title": "Find alternatives for small portfolio",
                "query": "What benchmarks work for a $100,000 portfolio?"
            },
            {
                "title": "Compare minimums",
                "query": "Compare minimums for S&P 500, Russell 1000, and NASDAQ-100"
            }
        ],
        "benchmark_search": [
            {
                "title": "International options",
                "query": "Show me international benchmark options"
            },
            {
                "title": "Factor-based benchmarks",
                "query": "What factor-based benchmarks are available?"
            },
            {
                "title": "Technology sector",
                "query": "Find technology-focused benchmarks"
            }
        ]
    }
    return suggestions

@app.post("/api/search")
async def search_benchmarks_api(query: SearchQuery):
    """Search benchmarks based on query"""
    try:
        results = search_benchmarks(
            query=query.query,
            filters=query.filters,
            top_k=5
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/benchmarks")
async def list_benchmarks():
    """List all available benchmarks"""
    try:
        with open("config/benchmarks.json", "r") as f:
            data = json.load(f)
            benchmarks = data.get("benchmarks", [])
            # Return simplified list
            return {
                "benchmarks": [
                    {
                        "name": b["name"],
                        "account_minimum": b["account_minimum"],
                        "market_cap": b.get("market_cap", "N/A")
                    }
                    for b in benchmarks[:20]  # Limit to first 20 for performance
                ]
            }
    except Exception as e:
        logger.error(f"Error listing benchmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time chat
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    
    # Create session if doesn't exist
    if session_id not in sessions:
        sessions[session_id] = ChatSession(session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle ping messages
            if message_data.get("type") == "ping":
                await manager.send_message(json.dumps({"type": "pong"}), session_id)
                continue
            
            # Check if message key exists for chat messages
            if "message" not in message_data:
                logger.error(f"Invalid message format: {message_data}")
                continue
            
            # Send typing indicator
            await manager.send_message(json.dumps({
                "type": "typing",
                "status": "start"
            }), session_id)
            
            # Process message
            chat_message = ChatMessage(
                message=message_data["message"],
                session_id=session_id
            )
            
            # Get response
            response = await chat(chat_message)
            
            # Send response back
            await manager.send_message(json.dumps({
                "type": "message",
                "data": response.model_dump()
            }), session_id)
            
            # Stop typing indicator
            await manager.send_message(json.dumps({
                "type": "typing",
                "status": "stop"
            }), session_id)
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"Client {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)