#!/usr/bin/env python3
"""
Capture baseline performance metrics for Ben AI before refactoring
Run this to establish current performance benchmarks
"""
import asyncio
import time
import json
import sys
from performance_monitor import monitor

# Import the functions we want to measure
from chatbot_core import (
    get_minimum,
    search_benchmarks,
    search_by_characteristics,
    get_all_benchmarks,
    client,
    CHAT_MODEL,
    FUNCTIONS
)

print("🚀 Ben AI Performance Baseline Capture")
print("=" * 50)

# Test 1: Measure get_minimum performance
print("\n📏 Testing get_minimum()...")
@monitor.measure("get_minimum")
def test_get_minimum():
    return get_minimum("S&P 500")

# Test 2: Measure search_benchmarks performance  
print("\n🔍 Testing search_benchmarks()...")
@monitor.measure("search_benchmarks")
def test_search():
    return search_benchmarks("technology")

# Test 3: Measure search_by_characteristics performance
print("\n🎯 Testing search_by_characteristics()...")
@monitor.measure("search_by_characteristics")
def test_alternatives():
    return search_by_characteristics("Russell 2000", portfolio_size=250000)

# Test 4: Measure get_all_benchmarks performance
print("\n📋 Testing get_all_benchmarks()...")
@monitor.measure("get_all_benchmarks")
def test_get_all():
    return get_all_benchmarks(filters={"factor": True})

# Test 5: Measure OpenAI API call performance
print("\n🤖 Testing OpenAI API call...")
@monitor.measure("openai_chat_completion")
def test_openai():
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the minimum for S&P 500?"}
        ],
        functions=FUNCTIONS,
        function_call="auto",
        temperature=0.1,
        max_tokens=100
    )
    return completion

# Run the baseline tests
def run_baseline_capture():
    """Run all baseline measurements multiple times for accuracy"""
    
    print("\n🔄 Running baseline measurements (3 iterations for accuracy)...")
    
    for i in range(3):
        print(f"\n--- Iteration {i+1}/3 ---")
        
        try:
            # Test each function
            test_get_minimum()
            time.sleep(0.5)  # Small delay between tests
            
            test_search()
            time.sleep(0.5)
            
            test_alternatives()
            time.sleep(0.5)
            
            test_get_all()
            time.sleep(0.5)
            
            # Only test OpenAI once per iteration (to save API costs)
            if i == 0:  # First iteration only
                test_openai()
            
        except Exception as e:
            print(f"❌ Error during testing: {e}")
            continue
    
    # Display summary
    print("\n" + "=" * 50)
    monitor.print_summary()
    
    print("\n✅ Baseline capture complete!")
    print(f"📁 Metrics saved to: performance_metrics.json")
    print("\nYou can view the metrics by opening performance_metrics.json")
    print("or run: python -c \"import json; print(json.dumps(json.load(open('performance_metrics.json')), indent=2))\"")

if __name__ == "__main__":
    run_baseline_capture()