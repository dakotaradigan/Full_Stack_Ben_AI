"""Data models for chat sessions and usage tracking."""

import threading
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class UsageTracker:
    """Track API usage for rate limiting and cost monitoring."""
    requests_per_minute: defaultdict = field(default_factory=lambda: defaultdict(int))
    costs_per_hour: defaultdict = field(default_factory=lambda: defaultdict(float))
    total_tokens: int = 0
    total_cost: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add_request(self, max_requests_per_minute: int = 100) -> bool:
        """Add a request and check if rate limit is exceeded."""
        current_minute = datetime.now().replace(second=0, microsecond=0)
        with self.lock:
            self.requests_per_minute[current_minute] += 1
            # Clean old entries
            cutoff = current_minute - timedelta(minutes=1)
            self.requests_per_minute = defaultdict(int, {
                k: v for k, v in self.requests_per_minute.items() if k > cutoff
            })
            return sum(self.requests_per_minute.values()) <= max_requests_per_minute
    
    def add_cost(self, tokens: int, cost: float, max_cost_per_hour: float = 50.0) -> bool:
        """Add cost and check if hourly limit is exceeded."""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        with self.lock:
            self.costs_per_hour[current_hour] += cost
            self.total_tokens += tokens
            self.total_cost += cost
            # Clean old entries
            cutoff = current_hour - timedelta(hours=1)
            self.costs_per_hour = defaultdict(float, {
                k: v for k, v in self.costs_per_hour.items() if k > cutoff
            })
            return sum(self.costs_per_hour.values()) <= max_cost_per_hour


@dataclass
class ChatSession:
    """Represents a chat session with message history and metadata."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    session_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    response_count: int = 0
    total_tokens: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to the session."""
        message = {"role": role, "content": content, **kwargs}
        self.messages.append(message)
        self.last_activity = datetime.now()
        
    def get_age_minutes(self) -> float:
        """Get session age in minutes."""
        return (datetime.now() - self.start_time).total_seconds() / 60


@dataclass 
class CircuitBreaker:
    """Circuit breaker pattern for resilient API calls."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    failure_count: int = 0
    last_failure_time: float = 0
    state: str = "closed"  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        import time
        
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e