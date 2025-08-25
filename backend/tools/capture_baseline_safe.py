#!/usr/bin/env python3
"""
Safe baseline capture without requiring API keys
Tests local functions only to establish internal performance baselines
"""
import time
import json
from performance_monitor import monitor

print("🚀 Ben AI Performance Baseline Capture (Safe Mode)")
print("=" * 50)
print("Note: Testing local functions only (no API calls)")

# Test file loading and basic operations
@monitor.measure("json_file_load")
def test_json_load():
    """Test loading benchmark data"""
    with open("config/benchmarks.json", "r") as f:
        data = json.load(f)
    return len(data.get("benchmarks", []))

@monitor.measure("benchmark_fuzzy_matching")  
def test_fuzzy_matching():
    """Test fuzzy string matching performance"""
    # Simulate the fuzzy matching logic without importing chatbot_core
    test_names = [
        "S&P 500", "Russell 2000", "NASDAQ-100", "MSCI World", 
        "ESG Domestic", "Technology", "International"
    ]
    query = "sp500"
    
    # Simple fuzzy matching simulation
    matches = []
    for name in test_names:
        name_lower = name.lower().replace("&", "").replace("-", "").replace(" ", "")
        if query in name_lower or name_lower in query:
            matches.append(name)
    return matches

@monitor.measure("system_prompt_load")
def test_system_prompt_load():
    """Test system prompt file loading"""
    with open("config/system_prompt.txt", "r") as f:
        content = f.read()
    return len(content)

@monitor.measure("description_generation")
def test_description_utils():
    """Test description utilities if available"""
    try:
        from utils.description_utils import build_semantic_description
        # Test with sample benchmark data
        sample_benchmark = {
            "name": "S&P 500",
            "account_minimum": 250000,
            "market_cap": "Large Cap",
            "region": "US"
        }
        return build_semantic_description(sample_benchmark)
    except ImportError:
        return "Description utils not available"

# Run safe baseline tests
def run_safe_baseline():
    """Run baseline measurements that don't require API access"""
    
    print("\n🔄 Running safe baseline measurements (5 iterations)...")
    
    for i in range(5):
        print(f"\n--- Iteration {i+1}/5 ---")
        
        try:
            # Test file operations
            benchmark_count = test_json_load()
            print(f"   Loaded {benchmark_count} benchmarks")
            time.sleep(0.2)
            
            # Test string processing
            matches = test_fuzzy_matching()
            print(f"   Found {len(matches)} fuzzy matches")
            time.sleep(0.2)
            
            # Test system prompt
            prompt_size = test_system_prompt_load()
            print(f"   System prompt: {prompt_size} characters")
            time.sleep(0.2)
            
            # Test description utils
            desc_result = test_description_utils()
            print(f"   Description generation: {'✅' if desc_result != 'Description utils not available' else '⚠️ Skipped'}")
            time.sleep(0.2)
            
        except Exception as e:
            print(f"❌ Error during iteration {i+1}: {e}")
            continue
    
    # Display summary
    print("\n" + "=" * 50)
    monitor.print_summary()
    
    print("\n✅ Safe baseline capture complete!")
    print(f"📁 Metrics saved to: performance_metrics.json")
    print("\nThese measurements will help detect regressions in:")
    print("  • File I/O performance")
    print("  • String processing speed")
    print("  • Data loading efficiency")
    print("  • Basic computational operations")
    
    print(f"\n💡 To test full system with API calls, set your OpenAI API key and run:")
    print(f"   python capture_baseline.py")

if __name__ == "__main__":
    run_safe_baseline()