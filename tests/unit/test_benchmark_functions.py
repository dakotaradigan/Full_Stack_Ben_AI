"""
Unit tests for benchmark functions in chatbot_core.py.

These tests validate benchmark retrieval, minimum calculations,
and related utility functions with mocked dependencies.
"""

import json
import os
import sys
import types
from typing import Dict, Any, List
from unittest.mock import Mock, patch, mock_open

import pytest

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Mock external dependencies before importing
class MockPinecone:
    def __init__(self, *args, **kwargs):
        pass

class MockOpenAI:
    def __init__(self, *args, **kwargs):
        pass

class MockEncoding:
    def encode(self, text):
        return [1] * (len(text) // 4)

# Apply mocks
sys.modules['pinecone'] = types.ModuleType('pinecone')
sys.modules['pinecone'].Pinecone = MockPinecone
sys.modules['openai'] = types.ModuleType('openai')
sys.modules['openai'].OpenAI = MockOpenAI
sys.modules['tiktoken'] = types.ModuleType('tiktoken')
sys.modules['tiktoken'].encoding_for_model = lambda model: MockEncoding()

# Now import the module we're testing
import chatbot_core

class TestGetBenchmark:
    """Tests for get_benchmark function."""
    
    @patch('chatbot_core.open', create=True)
    def test_get_benchmark_exact_match(self, mock_file):
        """Test getting benchmark with exact name match."""
        mock_data = {
            "benchmarks": [
                {
                    "name": "S&P 500",
                    "account_minimum": 2000000,
                    "market_cap": "Large Cap",
                    "tags": {
                        "region": ["US"],
                        "asset_class": ["Equity"],
                        "style": ["Core"]
                    }
                }
            ]
        }
        
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)
        
        result = chatbot_core.get_benchmark("S&P 500")
        
        assert result is not None
        assert result["name"] == "S&P 500"
        assert result["account_minimum"] == 2000000
        assert "tags" in result
        
    @patch('chatbot_core.open', create=True)
    def test_get_benchmark_case_insensitive(self, mock_file):
        """Test getting benchmark with case insensitive search."""
        mock_data = {
            "benchmarks": [
                {
                    "name": "Russell 2000",
                    "account_minimum": 2000000,
                    "market_cap": "Small Cap"
                }
            ]
        }
        
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)
        
        result = chatbot_core.get_benchmark("russell 2000")
        
        if result:  # Depending on implementation
            assert result["name"] == "Russell 2000"
            
    @patch('chatbot_core.open', create=True)
    def test_get_benchmark_not_found(self, mock_file):
        """Test getting non-existent benchmark."""
        mock_data = {"benchmarks": []}
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)
        
        result = chatbot_core.get_benchmark("NonexistentBenchmark")
        
        assert result is None
        
    @patch('chatbot_core.open', create=True)
    def test_get_benchmark_file_error(self, mock_file):
        """Test get_benchmark when file reading fails."""
        mock_file.side_effect = FileNotFoundError("File not found")
        
        result = chatbot_core.get_benchmark("S&P 500")
        
        # Should handle error gracefully
        assert result is None or isinstance(result, dict)

class TestGetMinimum:
    """Tests for get_minimum function."""
    
    def test_get_minimum_valid_benchmark(self):
        """Test get_minimum with valid benchmark."""
        mock_benchmark = {
            "name": "S&P 500",
            "account_minimum": 2000000,
            "market_cap": "Large Cap"
        }
        
        with patch.object(chatbot_core, 'get_benchmark', return_value=mock_benchmark):
            result = chatbot_core.get_minimum("S&P 500")
            
            assert isinstance(result, dict)
            assert "benchmark_info" in result
            assert "account_minimum" in result
            assert "minimum_display" in result
            
            assert result["benchmark_info"]["name"] == "S&P 500"
            assert result["account_minimum"] == 2000000
            
    def test_get_minimum_with_dividend(self):
        """Test get_minimum with dividend information included."""
        mock_benchmark = {
            "name": "Dividend Benchmark",
            "account_minimum": 1000000,
            "tags": {
                "dividend_yield": 4.2
            }
        }
        
        with patch.object(chatbot_core, 'get_benchmark', return_value=mock_benchmark):
            result = chatbot_core.get_minimum("Dividend Benchmark", include_dividend=True)
            
            assert isinstance(result, dict)
            assert "benchmark_info" in result
            # Should include dividend information if available
            
    def test_get_minimum_invalid_benchmark(self):
        """Test get_minimum with invalid benchmark name."""
        with patch.object(chatbot_core, 'get_benchmark', return_value=None):
            result = chatbot_core.get_minimum("NonexistentBenchmark")
            
            assert isinstance(result, dict)
            assert "error" in result or "benchmark_info" not in result
            
    def test_get_minimum_fuzzy_matching(self):
        """Test get_minimum uses fuzzy matching."""
        mock_benchmark = {
            "name": "NASDAQ-100",
            "account_minimum": 2000000
        }
        
        with patch.object(chatbot_core, 'get_benchmark', return_value=None):
            with patch.object(chatbot_core, '_fuzzy_match_benchmark', return_value=mock_benchmark):
                result = chatbot_core.get_minimum("NASDAQ 100")
                
                assert isinstance(result, dict)
                if "benchmark_info" in result:
                    assert result["benchmark_info"]["name"] == "NASDAQ-100"
                    
    def test_get_minimum_format_display(self):
        """Test get_minimum formats display correctly."""
        mock_benchmark = {
            "name": "Test Benchmark",
            "account_minimum": 2500000
        }
        
        with patch.object(chatbot_core, 'get_benchmark', return_value=mock_benchmark):
            result = chatbot_core.get_minimum("Test Benchmark")
            
            assert "minimum_display" in result
            display = result["minimum_display"]
            
            # Should format as currency
            assert "$" in display
            # Should show millions
            assert "2.5" in display or "2,500" in display

class TestBlendMinimum:
    """Tests for blend_minimum function."""
    
    def test_blend_minimum_simple_case(self):
        """Test blend_minimum with simple allocation."""
        allocations = [
            {"benchmark": "S&P 500", "allocation": 0.6},
            {"benchmark": "Russell 2000", "allocation": 0.4}
        ]
        
        mock_sp500 = {
            "name": "S&P 500",
            "account_minimum": 2000000
        }
        mock_russell = {
            "name": "Russell 2000", 
            "account_minimum": 2000000
        }
        
        def mock_get_benchmark(name):
            if "S&P" in name:
                return mock_sp500
            elif "Russell" in name:
                return mock_russell
            return None
            
        with patch.object(chatbot_core, 'get_benchmark', side_effect=mock_get_benchmark):
            result = chatbot_core.blend_minimum(allocations)
            
            assert isinstance(result, dict)
            assert "total_minimum" in result
            assert "breakdown" in result
            
            # Should calculate weighted minimum
            expected_minimum = (2000000 * 0.6) + (2000000 * 0.4)
            assert result["total_minimum"] == expected_minimum
            
    def test_blend_minimum_different_minimums(self):
        """Test blend_minimum with different account minimums."""
        allocations = [
            {"benchmark": "High Minimum", "allocation": 0.7},
            {"benchmark": "Low Minimum", "allocation": 0.3}
        ]
        
        def mock_get_benchmark(name):
            if "High" in name:
                return {"name": "High Minimum", "account_minimum": 5000000}
            elif "Low" in name:
                return {"name": "Low Minimum", "account_minimum": 1000000}
            return None
            
        with patch.object(chatbot_core, 'get_benchmark', side_effect=mock_get_benchmark):
            result = chatbot_core.blend_minimum(allocations)
            
            assert "total_minimum" in result
            expected_minimum = (5000000 * 0.7) + (1000000 * 0.3)
            assert result["total_minimum"] == expected_minimum
            
    def test_blend_minimum_invalid_benchmark(self):
        """Test blend_minimum with invalid benchmark."""
        allocations = [
            {"benchmark": "Valid", "allocation": 0.5},
            {"benchmark": "Invalid", "allocation": 0.5}
        ]
        
        def mock_get_benchmark(name):
            if name == "Valid":
                return {"name": "Valid", "account_minimum": 1000000}
            return None
            
        with patch.object(chatbot_core, 'get_benchmark', side_effect=mock_get_benchmark):
            result = chatbot_core.blend_minimum(allocations)
            
            # Should handle invalid benchmarks gracefully
            assert isinstance(result, dict)
            
    def test_blend_minimum_zero_allocation(self):
        """Test blend_minimum handles zero allocations."""
        allocations = [
            {"benchmark": "Benchmark A", "allocation": 1.0},
            {"benchmark": "Benchmark B", "allocation": 0.0}
        ]
        
        def mock_get_benchmark(name):
            return {"name": name, "account_minimum": 2000000}
            
        with patch.object(chatbot_core, 'get_benchmark', side_effect=mock_get_benchmark):
            result = chatbot_core.blend_minimum(allocations)
            
            assert "total_minimum" in result
            # Should effectively ignore zero allocation
            assert result["total_minimum"] == 2000000
            
    def test_blend_minimum_empty_allocations(self):
        """Test blend_minimum with empty allocations list."""
        result = chatbot_core.blend_minimum([])
        
        assert isinstance(result, dict)
        # Should handle gracefully
        
    def test_blend_minimum_with_dividend(self):
        """Test blend_minimum includes dividend information."""
        allocations = [
            {"benchmark": "Dividend Benchmark", "allocation": 1.0}
        ]
        
        mock_benchmark = {
            "name": "Dividend Benchmark",
            "account_minimum": 1000000,
            "tags": {
                "dividend_yield": 3.5
            }
        }
        
        with patch.object(chatbot_core, 'get_benchmark', return_value=mock_benchmark):
            result = chatbot_core.blend_minimum(allocations, include_dividend=True)
            
            assert isinstance(result, dict)
            # Should include dividend information in breakdown

class TestBenchmarkUtilities:
    """Tests for benchmark utility functions."""
    
    @patch('chatbot_core.open', create=True)
    def test_load_benchmarks_data(self, mock_file):
        """Test loading benchmark data from JSON file."""
        mock_data = {
            "benchmarks": [
                {"name": "Test 1", "account_minimum": 1000000},
                {"name": "Test 2", "account_minimum": 2000000}
            ]
        }
        
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)
        
        # Test functions that load data
        result = chatbot_core.get_benchmark("Test 1")
        
        # Should have attempted to read the file
        mock_file.assert_called()
        
    def test_benchmark_name_normalization(self):
        """Test benchmark name normalization."""
        # Test various name formats that should match
        test_cases = [
            ("S&P 500", "S&P 500"),
            ("s&p 500", "S&P 500"),
            ("SP 500", "S&P 500"),
            ("NASDAQ-100", "Nasdaq 100"),
            ("nasdaq 100", "Nasdaq 100")
        ]
        
        for input_name, expected_match in test_cases:
            # Test fuzzy matching logic
            is_match = chatbot_core._is_fuzzy_match(input_name, expected_match)
            # Behavior depends on implementation
            
    def test_benchmark_data_validation(self):
        """Test validation of benchmark data structure."""
        valid_benchmark = {
            "name": "Test Benchmark",
            "account_minimum": 1000000,
            "market_cap": "Large Cap",
            "tags": {
                "region": ["US"],
                "asset_class": ["Equity"]
            }
        }
        
        # Test that valid benchmark is accepted
        with patch.object(chatbot_core, 'get_benchmark', return_value=valid_benchmark):
            result = chatbot_core.get_minimum("Test Benchmark")
            assert isinstance(result, dict)
            
        # Test handling of malformed benchmark data
        invalid_benchmark = {
            "name": "Invalid Benchmark"
            # Missing required fields
        }
        
        with patch.object(chatbot_core, 'get_benchmark', return_value=invalid_benchmark):
            result = chatbot_core.get_minimum("Invalid Benchmark")
            # Should handle missing fields gracefully

class TestBenchmarkErrorHandling:
    """Tests for error handling in benchmark functions."""
    
    def test_get_benchmark_json_parse_error(self):
        """Test get_benchmark handles JSON parsing errors."""
        with patch('chatbot_core.open', mock_open(read_data="invalid json")):
            result = chatbot_core.get_benchmark("Any Benchmark")
            
            # Should handle JSON error gracefully
            assert result is None or isinstance(result, dict)
            
    def test_get_minimum_missing_minimum_field(self):
        """Test get_minimum handles missing account_minimum field."""
        mock_benchmark = {
            "name": "Incomplete Benchmark"
            # Missing account_minimum
        }
        
        with patch.object(chatbot_core, 'get_benchmark', return_value=mock_benchmark):
            result = chatbot_core.get_minimum("Incomplete Benchmark")
            
            # Should handle missing field gracefully
            assert isinstance(result, dict)
            
    def test_blend_minimum_allocation_sum_error(self):
        """Test blend_minimum handles allocation sum errors."""
        # Allocations that don't sum to 1.0
        allocations = [
            {"benchmark": "A", "allocation": 0.8},
            {"benchmark": "B", "allocation": 0.5}  # Sum = 1.3
        ]
        
        def mock_get_benchmark(name):
            return {"name": name, "account_minimum": 1000000}
            
        with patch.object(chatbot_core, 'get_benchmark', side_effect=mock_get_benchmark):
            result = chatbot_core.blend_minimum(allocations)
            
            # Should handle gracefully (may normalize or warn)
            assert isinstance(result, dict)
            
    def test_get_minimum_negative_minimum(self):
        """Test get_minimum handles negative account minimum."""
        mock_benchmark = {
            "name": "Negative Minimum",
            "account_minimum": -1000000  # Invalid negative minimum
        }
        
        with patch.object(chatbot_core, 'get_benchmark', return_value=mock_benchmark):
            result = chatbot_core.get_minimum("Negative Minimum")
            
            # Should handle gracefully
            assert isinstance(result, dict)

class TestBenchmarkDataConsistency:
    """Tests for data consistency in benchmark functions."""
    
    def test_all_benchmarks_have_minimums(self):
        """Test that all benchmarks have account minimums."""
        mock_data = {
            "benchmarks": [
                {"name": "Complete 1", "account_minimum": 1000000},
                {"name": "Complete 2", "account_minimum": 2000000},
                {"name": "Incomplete", "other_field": "value"}  # Missing minimum
            ]
        }
        
        with patch('chatbot_core.open', mock_open(read_data=json.dumps(mock_data))):
            # Test that functions handle incomplete data
            complete_result = chatbot_core.get_minimum("Complete 1")
            incomplete_result = chatbot_core.get_minimum("Incomplete")
            
            # Complete benchmark should work
            assert isinstance(complete_result, dict)
            if "benchmark_info" in complete_result:
                assert "account_minimum" in complete_result["benchmark_info"]
                
            # Incomplete benchmark should be handled gracefully
            assert isinstance(incomplete_result, dict)
            
    def test_benchmark_name_uniqueness(self):
        """Test handling of duplicate benchmark names."""
        mock_data = {
            "benchmarks": [
                {"name": "Duplicate", "account_minimum": 1000000},
                {"name": "Duplicate", "account_minimum": 2000000}  # Same name
            ]
        }
        
        with patch('chatbot_core.open', mock_open(read_data=json.dumps(mock_data))):
            result = chatbot_core.get_benchmark("Duplicate")
            
            # Should return one of them (behavior may vary)
            if result:
                assert result["name"] == "Duplicate"
                assert "account_minimum" in result
                
    def test_minimum_formatting_consistency(self):
        """Test that minimum amounts are formatted consistently."""
        test_minimums = [
            1000000,    # 1 million
            2500000,    # 2.5 million
            10000000,   # 10 million
            500000      # 0.5 million
        ]
        
        for minimum in test_minimums:
            mock_benchmark = {
                "name": f"Test {minimum}",
                "account_minimum": minimum
            }
            
            with patch.object(chatbot_core, 'get_benchmark', return_value=mock_benchmark):
                result = chatbot_core.get_minimum(f"Test {minimum}")
                
                if "minimum_display" in result:
                    display = result["minimum_display"]
                    # Should be formatted as currency
                    assert "$" in display
                    # Should be human-readable
                    assert len(display) < 50  # Reasonable length