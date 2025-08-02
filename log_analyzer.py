#!/usr/bin/env python3
"""
Chatbot Log Analysis Script
Monitors usage, costs, security events, and performance from chatbot.log
"""

import re
import json
import argparse
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os

@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    logger: str
    message: str
    raw_line: str

@dataclass
class SessionStats:
    session_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_tokens: int = 0
    total_cost: float = 0.0
    queries: int = 0
    function_calls: List[str] = field(default_factory=list)
    security_events: int = 0
    errors: int = 0
    
    @property
    def duration_minutes(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return 0.0

@dataclass  
class SecurityEvent:
    timestamp: datetime
    event_type: str
    details: str
    severity: str = "MEDIUM"

@dataclass
class UsageStats:
    total_sessions: int = 0
    total_queries: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_tokens_per_query: float = 0.0
    avg_cost_per_query: float = 0.0
    avg_session_duration: float = 0.0
    most_used_functions: Dict[str, int] = field(default_factory=dict)
    security_events: List[SecurityEvent] = field(default_factory=list)
    error_count: int = 0
    peak_hour: str = ""
    sessions_by_hour: Dict[int, int] = field(default_factory=dict)

class LogAnalyzer:
    def __init__(self, log_file: str = "chatbot.log"):
        self.log_file = log_file
        self.entries: List[LogEntry] = []
        self.sessions: Dict[str, SessionStats] = {}
        self.security_events: List[SecurityEvent] = []
        
    def parse_log_file(self) -> bool:
        """Parse the log file and extract structured data."""
        if not os.path.exists(self.log_file):
            print(f"❌ Log file '{self.log_file}' not found")
            return False
            
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            print(f"📁 Parsing {len(lines)} log entries from {self.log_file}")
            
            for line in lines:
                entry = self._parse_log_line(line.strip())
                if entry:
                    self.entries.append(entry)
                    self._process_entry(entry)
                    
            print(f"✅ Parsed {len(self.entries)} valid log entries")
            return True
            
        except Exception as e:
            print(f"❌ Error parsing log file: {e}")
            return False
    
    def _parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a single log line into a LogEntry object."""
        # Format: 2024-01-15 10:30:15 - chatbot_enhanced - INFO - Message
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ([^-]+) - ([^-]+) - (.+)'
        match = re.match(pattern, line)
        
        if not match:
            return None
            
        try:
            timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
            return LogEntry(
                timestamp=timestamp,
                level=match.group(3).strip(),
                logger=match.group(2).strip(),
                message=match.group(4).strip(),
                raw_line=line
            )
        except ValueError:
            return None
    
    def _process_entry(self, entry: LogEntry):
        """Process a log entry to extract relevant information."""
        message = entry.message
        
        # Session tracking
        if "Session" in message and "ended normally" in message:
            session_match = re.search(r'Session (\w+) ended normally', message)
            if session_match:
                session_id = session_match.group(1)
                if session_id not in self.sessions:
                    self.sessions[session_id] = SessionStats(session_id)
                self.sessions[session_id].end_time = entry.timestamp
        
        # Token and cost tracking
        if "API call:" in message:
            api_match = re.search(r'API call: (\d+) tokens, \$([0-9.]+), total: \$([0-9.]+)', message)
            if api_match:
                tokens = int(api_match.group(1))
                cost = float(api_match.group(2))
                total_cost = float(api_match.group(3))
                
                # Find current session (approximate by timestamp)
                current_session = self._get_current_session(entry.timestamp)
                if current_session:
                    current_session.total_tokens += tokens
                    current_session.total_cost = total_cost
        
        # Function call tracking
        if "Calling function:" in message:
            func_match = re.search(r'Calling function: (\w+)', message)
            if func_match:
                func_name = func_match.group(1)
                current_session = self._get_current_session(entry.timestamp)
                if current_session:
                    current_session.function_calls.append(func_name)
                    current_session.queries += 1
        
        # Security event tracking
        if any(keyword in message.lower() for keyword in [
            "malicious input detected", "rate limit", "cost limit", "circuit breaker", "filtered"
        ]):
            severity = "HIGH" if any(high in message.lower() for high in ["cost limit", "circuit breaker"]) else "MEDIUM"
            
            event = SecurityEvent(
                timestamp=entry.timestamp,
                event_type=self._classify_security_event(message),
                details=message,
                severity=severity
            )
            self.security_events.append(event)
            
            current_session = self._get_current_session(entry.timestamp)
            if current_session:
                current_session.security_events += 1
        
        # Error tracking
        if entry.level in ["ERROR", "WARNING"]:
            current_session = self._get_current_session(entry.timestamp)
            if current_session:
                current_session.errors += 1
    
    def _get_current_session(self, timestamp: datetime) -> Optional[SessionStats]:
        """Find the session that was active at the given timestamp."""
        # Simple heuristic: find session within 1 hour of timestamp
        for session in self.sessions.values():
            if session.start_time and abs((timestamp - session.start_time).total_seconds()) < 3600:
                return session
        return None
    
    def _classify_security_event(self, message: str) -> str:
        """Classify the type of security event."""
        message_lower = message.lower()
        if "malicious input" in message_lower or "filtered" in message_lower:
            return "PROMPT_INJECTION_ATTEMPT"
        elif "rate limit" in message_lower:
            return "RATE_LIMIT_EXCEEDED"
        elif "cost limit" in message_lower:
            return "COST_LIMIT_EXCEEDED"
        elif "circuit breaker" in message_lower:
            return "CIRCUIT_BREAKER_TRIGGERED"
        else:
            return "SECURITY_EVENT"
    
    def generate_usage_stats(self) -> UsageStats:
        """Generate comprehensive usage statistics."""
        stats = UsageStats()
        
        # Basic counts
        stats.total_sessions = len(self.sessions)
        stats.security_events = self.security_events
        stats.error_count = sum(1 for entry in self.entries if entry.level == "ERROR")
        
        # Session-based stats
        valid_sessions = [s for s in self.sessions.values() if s.queries > 0]
        
        if valid_sessions:
            stats.total_queries = sum(s.queries for s in valid_sessions)
            stats.total_tokens = sum(s.total_tokens for s in valid_sessions)
            stats.total_cost = sum(s.total_cost for s in valid_sessions)
            
            if stats.total_queries > 0:
                stats.avg_tokens_per_query = stats.total_tokens / stats.total_queries
                stats.avg_cost_per_query = stats.total_cost / stats.total_queries
            
            durations = [s.duration_minutes for s in valid_sessions if s.duration_minutes > 0]
            if durations:
                stats.avg_session_duration = sum(durations) / len(durations)
        
        # Function usage stats
        all_functions = []
        for session in valid_sessions:
            all_functions.extend(session.function_calls)
        stats.most_used_functions = dict(Counter(all_functions).most_common(10))
        
        # Peak usage analysis
        hours_usage = defaultdict(int)
        for entry in self.entries:
            if "Calling function:" in entry.message:
                hours_usage[entry.timestamp.hour] += 1
        
        if hours_usage:
            peak_hour_num = max(hours_usage, key=hours_usage.get)
            stats.peak_hour = f"{peak_hour_num:02d}:00"
            stats.sessions_by_hour = dict(hours_usage)
        
        return stats
    
    def print_summary_report(self):
        """Print a comprehensive summary report."""
        stats = self.generate_usage_stats()
        
        print("\n" + "="*60)
        print("🤖 CHATBOT USAGE ANALYSIS REPORT")  
        print("="*60)
        
        # Overall Stats
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Sessions: {stats.total_sessions}")
        print(f"   Total Queries: {stats.total_queries}")
        print(f"   Total Tokens Used: {stats.total_tokens:,}")
        print(f"   Total Cost: ${stats.total_cost:.2f}")
        print(f"   Average Session Duration: {stats.avg_session_duration:.1f} minutes")
        
        if stats.total_queries > 0:
            print(f"   Average Tokens per Query: {stats.avg_tokens_per_query:.0f}")
            print(f"   Average Cost per Query: ${stats.avg_cost_per_query:.4f}")
        
        # Function Usage
        if stats.most_used_functions:
            print(f"\n🔧 MOST USED FUNCTIONS:")
            for func, count in stats.most_used_functions.items():
                percentage = (count / stats.total_queries * 100) if stats.total_queries > 0 else 0
                print(f"   {func}: {count} times ({percentage:.1f}%)")
        
        # Peak Usage
        if stats.peak_hour:
            print(f"\n⏰ PEAK USAGE:")
            print(f"   Peak Hour: {stats.peak_hour}")
            print(f"   Hourly Distribution:")
            for hour in sorted(stats.sessions_by_hour.keys()):
                count = stats.sessions_by_hour[hour]
                bar = "█" * min(count, 20)
                print(f"   {hour:02d}:00 [{count:2d}] {bar}")
        
        # Security Events
        if stats.security_events:
            print(f"\n🛡️ SECURITY EVENTS ({len(stats.security_events)} total):")
            event_types = Counter(event.event_type for event in stats.security_events)
            for event_type, count in event_types.most_common():
                print(f"   {event_type}: {count} events")
            
            # Recent security events
            recent_events = sorted(stats.security_events, key=lambda x: x.timestamp, reverse=True)[:5]
            if recent_events:
                print(f"\n   Recent Security Events:")
                for event in recent_events:
                    print(f"   {event.timestamp.strftime('%Y-%m-%d %H:%M')} - {event.event_type}")
                    print(f"     └─ {event.details[:80]}{'...' if len(event.details) > 80 else ''}")
        
        # Error Summary
        if stats.error_count > 0:
            print(f"\n❌ ERRORS: {stats.error_count} total errors logged")
            
            # Get recent errors
            recent_errors = [e for e in self.entries if e.level == "ERROR"][-5:]
            if recent_errors:
                print("   Recent Errors:")
                for error in recent_errors:
                    print(f"   {error.timestamp.strftime('%Y-%m-%d %H:%M')} - {error.message[:60]}...")
        
        print("\n" + "="*60)
    
    def print_cost_analysis(self, days: int = 7):
        """Print detailed cost analysis for the last N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_entries = [e for e in self.entries if e.timestamp >= cutoff_date]
        
        print(f"\n💰 COST ANALYSIS (Last {days} days)")
        print("-" * 40)
        
        # Extract cost data
        daily_costs = defaultdict(float)
        daily_tokens = defaultdict(int)
        
        for entry in recent_entries:
            if "API call:" in entry.message:
                match = re.search(r'API call: (\d+) tokens, \$([0-9.]+)', entry.message)
                if match:
                    day = entry.timestamp.date()
                    tokens = int(match.group(1))
                    cost = float(match.group(2))
                    daily_costs[day] += cost
                    daily_tokens[day] += tokens
        
        total_recent_cost = sum(daily_costs.values())
        total_recent_tokens = sum(daily_tokens.values())
        
        print(f"Total Cost ({days} days): ${total_recent_cost:.2f}")
        print(f"Total Tokens ({days} days): {total_recent_tokens:,}")
        print(f"Average Daily Cost: ${total_recent_cost/days:.2f}")
        
        if daily_costs:
            print(f"\nDaily Breakdown:")
            for day in sorted(daily_costs.keys(), reverse=True):
                cost = daily_costs[day]
                tokens = daily_tokens[day]
                print(f"  {day}: ${cost:.2f} ({tokens:,} tokens)")
    
    def export_to_json(self, output_file: str = "usage_report.json"):
        """Export analysis results to JSON file."""
        stats = self.generate_usage_stats()
        
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "log_file": self.log_file,
            "total_entries": len(self.entries),
            "analysis_period": {
                "start": min(e.timestamp for e in self.entries).isoformat() if self.entries else None,
                "end": max(e.timestamp for e in self.entries).isoformat() if self.entries else None,
            },
            "usage_stats": {
                "total_sessions": stats.total_sessions,
                "total_queries": stats.total_queries,
                "total_tokens": stats.total_tokens,
                "total_cost": stats.total_cost,
                "avg_tokens_per_query": stats.avg_tokens_per_query,
                "avg_cost_per_query": stats.avg_cost_per_query,
                "avg_session_duration": stats.avg_session_duration,
                "most_used_functions": stats.most_used_functions,
                "peak_hour": stats.peak_hour,
                "sessions_by_hour": stats.sessions_by_hour,
            },
            "security_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.event_type,
                    "severity": event.severity,
                    "details": event.details
                }
                for event in stats.security_events
            ],
            "error_count": stats.error_count,
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"📁 Report exported to {output_file}")
        except Exception as e:
            print(f"❌ Failed to export report: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze chatbot usage logs")
    parser.add_argument("--log-file", "-f", default="chatbot.log", 
                       help="Path to log file (default: chatbot.log)")
    parser.add_argument("--cost-days", "-d", type=int, default=7,
                       help="Days for cost analysis (default: 7)")
    parser.add_argument("--export", "-e", action="store_true",
                       help="Export results to JSON file")
    parser.add_argument("--export-file", default="usage_report.json",
                       help="JSON export filename (default: usage_report.json)")
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(args.log_file)
    
    if not analyzer.parse_log_file():
        return 1
    
    analyzer.print_summary_report()
    analyzer.print_cost_analysis(args.cost_days)
    
    if args.export:
        analyzer.export_to_json(args.export_file)
    
    return 0

if __name__ == "__main__":
    exit(main())