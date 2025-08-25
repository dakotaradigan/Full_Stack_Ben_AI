#!/usr/bin/env python3
"""
Quick performance check for Ben AI
View current performance metrics and baselines
"""
import json
import os
from datetime import datetime, timedelta

def check_performance():
    """Display current performance metrics in a readable format"""
    
    if not os.path.exists("performance_metrics.json"):
        print("❌ No performance metrics found.")
        print("Run: python capture_baseline.py to capture baseline metrics first.")
        return
    
    with open("performance_metrics.json", 'r') as f:
        data = json.load(f)
    
    print("📊 Ben AI Performance Report")
    print("=" * 60)
    
    # Show baselines
    if data.get("baselines"):
        print("\n🎯 Current Baselines:")
        print("-" * 40)
        for operation, stats in data["baselines"].items():
            status = "🟢" if stats["average_ms"] < 500 else "🟡" if stats["average_ms"] < 1000 else "🔴"
            print(f"{status} {operation}:")
            print(f"    Average: {stats['average_ms']}ms")
            print(f"    Range: {stats['min_ms']}ms - {stats['max_ms']}ms")
            print(f"    Samples: {stats['sample_size']}")
            print()
    
    # Show recent measurements (last 10)
    recent_measurements = data.get("measurements", [])[-10:]
    if recent_measurements:
        print("\n📈 Recent Measurements (last 10):")
        print("-" * 40)
        for measurement in recent_measurements:
            timestamp = datetime.fromisoformat(measurement["timestamp"]).strftime("%H:%M:%S")
            status = "✅" if measurement["success"] else "❌"
            duration = measurement["duration_ms"]
            operation = measurement["operation"]
            
            # Color coding for performance
            if duration < 200:
                speed_icon = "🚀"  # Very fast
            elif duration < 500:
                speed_icon = "✅"  # Good
            elif duration < 1000:
                speed_icon = "🟡"  # Okay
            else:
                speed_icon = "🔴"  # Slow
                
            print(f"{timestamp} {status} {speed_icon} {operation}: {duration}ms")
    
    # Performance trend analysis
    if len(data.get("measurements", [])) >= 5:
        measurements = data["measurements"]
        
        # Group by operation for trend analysis
        operations = {}
        for m in measurements:
            if m["success"]:  # Only successful measurements
                op = m["operation"]
                if op not in operations:
                    operations[op] = []
                operations[op].append(m["duration_ms"])
        
        print("\n📊 Performance Trends:")
        print("-" * 40)
        for op, times in operations.items():
            if len(times) >= 3:  # Need at least 3 measurements for trend
                recent_avg = sum(times[-3:]) / 3  # Last 3 measurements
                early_avg = sum(times[:3]) / 3    # First 3 measurements
                
                if recent_avg < early_avg * 0.9:  # 10% improvement
                    trend = "📈 Improving"
                elif recent_avg > early_avg * 1.1:  # 10% degradation
                    trend = "📉 Degrading"
                else:
                    trend = "➡️  Stable"
                
                change_pct = ((recent_avg - early_avg) / early_avg) * 100
                print(f"{trend} {op}: {change_pct:+.1f}%")
    
    print("\n" + "=" * 60)
    total_measurements = len(data.get("measurements", []))
    print(f"Total measurements captured: {total_measurements}")
    
    if total_measurements > 0:
        last_measurement = datetime.fromisoformat(data["measurements"][-1]["timestamp"])
        time_since = datetime.now() - last_measurement
        if time_since < timedelta(hours=1):
            print(f"Last measurement: {time_since.seconds//60} minutes ago")
        else:
            print(f"Last measurement: {last_measurement.strftime('%Y-%m-%d %H:%M')}")

def check_regression():
    """Check if performance has regressed compared to baseline"""
    if not os.path.exists("performance_metrics.json"):
        print("❌ No metrics file found")
        return
    
    with open("performance_metrics.json", 'r') as f:
        data = json.load(f)
    
    recent_measurements = data.get("measurements", [])[-5:]  # Last 5 measurements
    baselines = data.get("baselines", {})
    
    print("\n🔍 Regression Check:")
    print("-" * 30)
    
    regressions_found = False
    
    for measurement in recent_measurements:
        if not measurement["success"]:
            continue
            
        operation = measurement["operation"]
        current_time = measurement["duration_ms"]
        
        if operation in baselines:
            baseline_avg = baselines[operation]["average_ms"]
            if current_time > baseline_avg * 1.5:  # 50% slower
                print(f"🚨 REGRESSION: {operation}")
                print(f"   Current: {current_time}ms vs Baseline: {baseline_avg}ms")
                print(f"   {((current_time/baseline_avg - 1) * 100):+.1f}% change")
                regressions_found = True
    
    if not regressions_found:
        print("✅ No significant performance regressions detected")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "regression":
        check_regression()
    else:
        check_performance()