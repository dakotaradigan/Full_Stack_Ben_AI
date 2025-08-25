"""Chat service for OpenAI integration, function calling, and session management."""

import json
import logging
import os
import re
import time
from typing import List, Dict, Any, Optional, Callable

from openai import OpenAI

from services.base_service import BaseService
from models.chat_models import ChatSession, UsageTracker, CircuitBreaker
from utils.token_utils import num_tokens_from_messages, trim_history, estimate_cost
from utils.security_utils import validate_response_security, sanitize_input

logger = logging.getLogger(__name__)


class ChatService(BaseService):
    """Service for chat operations, OpenAI integration, and function calling."""
    
    def __init__(self):
        super().__init__()
        self.validate_config()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Configuration
        self.chat_model = "gpt-4o"
        self.max_tokens_per_request = 12000
        self.max_requests_per_minute = 100
        self.max_cost_per_hour = 50.0
        
        # Load system prompt
        with open("config/system_prompt.txt", "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
        
        # Extract disclaimer text
        match = re.search(
            r"disclaimer every 3-4 interactions:\s*\*?[\"\u201c](.+?)[\"\u201d]\*?",
            self.system_prompt,
            flags=re.IGNORECASE,
        )
        self.disclaimer_text = (
            match.group(1).strip()
            if match
            else "***This assistant provides benchmark eligibility guidance only. No investment advice or account approval authority. Ben gets things wrong too so check with your sales rep if you have questions. Also use our on-demand transition analysis tool for further analysis.***"
        )
        self.disclaimer_frequency = 3
        
        # Usage tracking and circuit breaker
        self.usage_tracker = UsageTracker()
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        # Function definitions for OpenAI function calling
        self.functions = self._load_function_definitions()
    
    def _load_function_definitions(self) -> List[Dict[str, Any]]:
        """Load function definitions for OpenAI function calling."""
        return [
            {
                "name": "search_benchmarks",
                "description": "Semantic search for complex criteria and natural language queries (e.g., 'Find good ESG technology benchmarks'). Use for multi-criteria searches with descriptive text and find/search intent patterns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language search query"},
                        "top_k": {"type": "integer", "description": "Number of results", "default": 5},
                        "filters": {"type": "object", "description": "Optional filters (region, asset_class, style, etc.)"},
                        "include_dividend": {"type": "boolean", "description": "Include dividend yield data", "default": False},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_minimum",
                "description": "Get minimum investment amount for a specific benchmark by exact name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Exact benchmark name"},
                        "include_dividend": {"type": "boolean", "description": "Include dividend yield data", "default": False},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "blend_minimum",
                "description": "Calculate blended minimum for portfolio allocation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "allocations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "weight": {"type": "number"}
                                },
                                "required": ["name", "weight"]
                            }
                        },
                        "include_dividend": {"type": "boolean", "default": False},
                    },
                    "required": ["allocations"],
                },
            },
            {
                "name": "search_by_characteristics",
                "description": "Find alternatives to a reference benchmark with similar characteristics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reference_benchmark": {"type": "string", "description": "Reference benchmark name"},
                        "portfolio_size": {"type": "number", "description": "Portfolio size constraint"},
                        "top_k": {"type": "integer", "description": "Number of results", "default": 5},
                        "include_dividend": {"type": "boolean", "default": False},
                    },
                    "required": ["reference_benchmark"],
                },
            },
            {
                "name": "get_all_benchmarks",
                "description": "Retrieve comprehensive lists by exact category filtering (e.g., 'What factor benchmarks are available?'). Use for listing intent patterns with 'what', 'show me', 'list' queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_dividend": {"type": "boolean", "default": False},
                        "filters": {"type": "object", "description": "Category filters (region, asset_class, style, etc.)"},
                    },
                },
            },
        ]
    
    def create_session(self, session_id: str) -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(session_id=session_id)
        session.add_message("system", self.system_prompt)
        return session
    
    def process_message(
        self,
        session: ChatSession,
        message: str,
        function_registry: Dict[str, Callable] = None
    ) -> Dict[str, Any]:
        """Process a chat message with function calling support."""
        try:
            # Rate limiting check
            if not self.usage_tracker.add_request(self.max_requests_per_minute):
                return {
                    "error": "Rate limit exceeded. Please try again in a moment.",
                    "retry_after": 60
                }
            
            # Sanitize input
            sanitized_message = sanitize_input(message)
            if sanitized_message != message:
                logger.info("Input was sanitized for security")
            
            # Add user message to session
            session.add_message("user", sanitized_message)
            
            # Trim history if needed
            messages_copy = session.messages.copy()
            was_trimmed = trim_history(messages_copy, limit=self.max_tokens_per_request)
            if was_trimmed:
                logger.info("Message history was trimmed due to token limit")
            
            # Count tokens
            total_tokens = num_tokens_from_messages(messages_copy, self.chat_model)
            estimated_cost = estimate_cost(total_tokens, self.chat_model)
            
            # Cost check
            if not self.usage_tracker.add_cost(total_tokens, estimated_cost, self.max_cost_per_hour):
                return {
                    "error": "Hourly cost limit exceeded. Please try again later.",
                    "retry_after": 3600
                }
            
            # Make OpenAI API call with function calling
            response_data = self._call_openai_with_functions(
                messages_copy, function_registry
            )
            
            # Update session
            session.messages = messages_copy
            session.response_count += 1
            session.total_tokens += total_tokens
            
            # Add disclaimer if needed
            if self._should_add_disclaimer(session):
                response_data["message"] += f"\n\n{self.disclaimer_text}"
            
            return response_data
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}", exc_info=True)
            return {
                "error": "I'm experiencing technical difficulties. Please try again or contact support.",
                "message": "Sorry, I encountered an error processing your request."
            }
    
    def _call_openai_with_functions(
        self, 
        messages: List[Dict[str, Any]],
        function_registry: Dict[str, Callable] = None
    ) -> Dict[str, Any]:
        """Make OpenAI API call with iterative function calling support."""
        function_calls_made = []
        function_calls_successful = False
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            try:
                # Make API call
                response = self.circuit_breaker.call(
                    self.client.chat.completions.create,
                    model=self.chat_model,
                    messages=messages,
                    functions=self.functions,
                    function_call="auto",
                    temperature=0.1,
                    max_tokens=4000,
                    timeout=30.0
                )
                
                message = response.choices[0].message
                
                # Handle function calls
                if message.function_call:
                    function_name = message.function_call.name
                    
                    try:
                        function_args = json.loads(message.function_call.arguments)
                    except json.JSONDecodeError:
                        function_args = {}
                    
                    # Check for duplicate function calls
                    call_signature = (function_name, str(sorted(function_args.items())))
                    if call_signature in function_calls_made:
                        logger.warning(f"Duplicate function call detected: {function_name}")
                        break
                    
                    function_calls_made.append(call_signature)
                    
                    # Execute function
                    if function_registry and function_name in function_registry:
                        function_result = function_registry[function_name](function_args)
                        function_calls_successful = True
                    else:
                        function_result = {"error": f"Function {function_name} not available"}
                    
                    # Add function call and result to messages
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": function_name,
                            "arguments": message.function_call.arguments
                        }
                    })
                    messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(function_result)
                    })
                    
                    iteration += 1
                    continue
                
                else:
                    # No function call, we have the final response
                    final_response = message.content or ""
                    
                    # Validate response security
                    validated_response = validate_response_security(
                        final_response, function_calls_successful
                    )
                    
                    # Add assistant message to conversation
                    messages.append({
                        "role": "assistant",
                        "content": validated_response
                    })
                    
                    return {
                        "message": validated_response,
                        "function_calls_made": len(function_calls_made),
                        "function_calls_successful": function_calls_successful,
                        "iterations": iteration
                    }
                    
            except Exception as e:
                logger.error(f"OpenAI API call failed (iteration {iteration}): {e}")
                if iteration == 0:  # First call failed
                    raise e
                else:  # Later iterations failed, return what we have
                    break
        
        # If we exit the loop without a normal response
        fallback_message = self._get_fallback_response()
        messages.append({
            "role": "assistant", 
            "content": fallback_message
        })
        
        return {
            "message": fallback_message,
            "function_calls_made": len(function_calls_made),
            "function_calls_successful": function_calls_successful,
            "iterations": iteration,
            "timeout": True
        }
    
    def call_function(self, name: str, arguments: Dict[str, Any], function_registry: Dict[str, Callable] = None) -> Dict[str, Any]:
        """Call a specific function by name."""
        try:
            logger.info(f"Calling function: {name} with args: {arguments}")
            
            # Validate function name
            function_names = [f["name"] for f in self.functions]
            if name not in function_names:
                return {"error": f"Unknown function: {name}"}
            
            # Call function through registry
            if function_registry and name in function_registry:
                return function_registry[name](arguments)
            else:
                return {"error": f"Function {name} not available in registry"}
                
        except Exception as e:
            logger.error(f"Function call failed for {name}: {e}", exc_info=True)
            return {"error": f"Function {name} failed due to system error"}
    
    def _should_add_disclaimer(self, session: ChatSession) -> bool:
        """Check if disclaimer should be added to response."""
        return (session.response_count % self.disclaimer_frequency == 0 and 
                session.response_count > 0)
    
    def _get_fallback_response(self) -> str:
        """Get a random fallback response for errors."""
        import random
        fallback_responses = [
            "I'm experiencing technical difficulties accessing the benchmark database. Please try again in a moment, or contact your Sales representative for immediate assistance.",
            "I'm unable to process that request right now due to system issues. For urgent benchmark questions, please reach out to your Sales team directly.",
            "The benchmark lookup system is temporarily unavailable. I apologize for the inconvenience. Your Sales representative can provide immediate assistance with benchmark eligibility questions.",
        ]
        return random.choice(fallback_responses)
    
    def get_session_stats(self, session: ChatSession) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": session.session_id,
            "message_count": len(session.messages),
            "response_count": session.response_count,
            "total_tokens": session.total_tokens,
            "session_age_minutes": session.get_age_minutes(),
            "last_activity": session.last_activity.isoformat()
        }