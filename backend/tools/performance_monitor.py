"""
Minimal Performance Monitoring for Ben AI
Captures baseline metrics before refactoring to ensure no performance regression
"""
import time
import json
import os
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Callable

class PerformanceMonitor:
    """Simple performance monitor that logs metrics to a JSON file"""
    
    def __init__(self, log_file: str = "performance_metrics.json"):
        self.log_file = log_file
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict:
        """Load existing metrics or create new file"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {"baselines": {}, "measurements": []}
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def measure(self, operation_name: str):
        """Decorator to measure function execution time"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.record_metric(operation_name, duration, success=True)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.record_metric(operation_name, duration, success=False, error=str(e))
                    raise
            return wrapper
        return decorator
    
    def record_metric(self, operation: str, duration: float, success: bool = True, error: str = None):
        """Record a single metric measurement"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration_ms": round(duration * 1000, 2),  # Convert to milliseconds
            "success": success
        }
        if error:
            metric["error"] = error
        
        self.metrics["measurements"].append(metric)
        
        # Update baseline (average of last 10 successful measurements)
        successful_measurements = [
            m["duration_ms"] for m in self.metrics["measurements"]
            if m["operation"] == operation and m["success"]
        ][-10:]
        
        if successful_measurements:
            self.metrics["baselines"][operation] = {
                "average_ms": round(sum(successful_measurements) / len(successful_measurements), 2),
                "min_ms": round(min(successful_measurements), 2),
                "max_ms": round(max(successful_measurements), 2),
                "sample_size": len(successful_measurements)
            }
        
        self._save_metrics()
        
        # Print to console for immediate feedback
        status = "✅" if success else "❌"
        print(f"{status} {operation}: {metric['duration_ms']}ms")
        
        # Alert if performance degrades significantly (>50% slower than baseline)
        if operation in self.metrics["baselines"] and success:
            baseline = self.metrics["baselines"][operation]["average_ms"]
            if metric["duration_ms"] > baseline * 1.5:
                print(f"⚠️  WARNING: {operation} is {round((metric['duration_ms']/baseline - 1) * 100)}% slower than baseline!")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "baselines": self.metrics["baselines"],
            "total_measurements": len(self.metrics["measurements"]),
            "operations_tracked": list(self.metrics["baselines"].keys())
        }
    
    def print_summary(self):
        """Print a readable summary of performance metrics"""
        print("\n📊 Performance Summary")
        print("=" * 50)
        for operation, stats in self.metrics["baselines"].items():
            print(f"\n{operation}:")
            print(f"  Average: {stats['average_ms']}ms")
            print(f"  Range: {stats['min_ms']}ms - {stats['max_ms']}ms")
            print(f"  Samples: {stats['sample_size']}")

# Global instance
monitor = PerformanceMonitor()