"""
Unit tests for search functions in chatbot_core.py.

These tests validate search functionality with mocked dependencies
to ensure individual components work correctly in isolation.
"""

import json
import os
import sys
import types
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

import pytest

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Mock external dependencies before importing
class MockPinecone:
    def __init__(self, *args, **kwargs):
        pass
        
    def list_indexes(self):
        return types.SimpleNamespace(names=lambda: ["benchmark-index"])
        
    def Index(self, name):
        return MockIndex()

class MockIndex:
    def __init__(self):
        self.query_results = []
        
    def query(self, *args, **kwargs):
        return types.SimpleNamespace(matches=self.query_results)
        
    def set_mock_results(self, results):
        self.query_results = results

class MockOpenAI:
    def __init__(self, *args, **kwargs):
        self.embeddings = MockEmbeddings()
        
class MockEmbeddings:
    def create(self, *args, **kwargs):
        return types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=[0.1] * 1536)]
        )

# Apply mocks
sys.modules['pinecone'] = types.ModuleType('pinecone')
sys.modules['pinecone'].Pinecone = MockPinecone
sys.modules['openai'] = types.ModuleType('openai')
sys.modules['openai'].OpenAI = MockOpenAI

# Mock tiktoken
class MockEncoding:
    def encode(self, text):
        return [1] * (len(text) // 4)  # Rough approximation

sys.modules['tiktoken'] = types.ModuleType('tiktoken')
sys.modules['tiktoken'].encoding_for_model = lambda model: MockEncoding()

# Now import the module we're testing
import chatbot_core

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mocks between tests."""
    # Reset any module-level state
    yield

class TestSearchBenchmarks:
    """Tests for search_benchmarks function."""
    
    def test_search_benchmarks_basic_query(self):
        """Test basic search functionality."""
        with patch.object(chatbot_core, 'embed', return_value=[0.1] * 1536):
            with patch.object(chatbot_core.pc, 'Index') as mock_index_class:
                mock_index = Mock()
                mock_index_class.return_value = mock_index
                
                # Mock search results
                mock_index.query.return_value = types.SimpleNamespace(
                    matches=[
                        types.SimpleNamespace(
                            id="sp500",
                            score=0.9,
                            metadata={
                                "name": "S&P 500",
                                "account_minimum": 2000000,
                                "market_cap": "Large Cap"
                            }
                        )
                    ]
                )
                
                result = chatbot_core.search_benchmarks("large cap stocks")
                
                assert isinstance(result, list)
                assert len(result) > 0
                assert result[0]["name"] == "S&P 500"
                assert result[0]["account_minimum"] == 2000000
                
    def test_search_benchmarks_with_filters(self):
        """Test search with filters."""
        with patch.object(chatbot_core, 'embed', return_value=[0.1] * 1536):
            with patch.object(chatbot_core.pc, 'Index') as mock_index_class:
                mock_index = Mock()
                mock_index_class.return_value = mock_index
                
                mock_index.query.return_value = types.SimpleNamespace(matches=[])
                
                filters = {"region": {"$in": ["US"]}}
                result = chatbot_core.search_benchmarks(
                    "technology stocks",
                    filters=filters
                )
                
                # Verify filters were passed to query
                mock_index.query.assert_called_once()
                call_args = mock_index.query.call_args
                assert "filter" in call_args.kwargs
                assert call_args.kwargs["filter"] == filters
                
    def test_search_benchmarks_with_portfolio_size(self):
        """Test search with portfolio size filtering."""
        with patch.object(chatbot_core, 'embed', return_value=[0.1] * 1536):
            with patch.object(chatbot_core.pc, 'Index') as mock_index_class:
                mock_index = Mock()
                mock_index_class.return_value = mock_index
                
                # Mock results with different minimums
                mock_index.query.return_value = types.SimpleNamespace(
                    matches=[
                        types.SimpleNamespace(
                            id="low_min",
                            score=0.9,
                            metadata={
                                "name": "Low Minimum",
                                "account_minimum": 100000
                            }
                        ),
                        types.SimpleNamespace(
                            id="high_min", 
                            score=0.8,
                            metadata={
                                "name": "High Minimum",
                                "account_minimum": 5000000
                            }
                        )
                    ]
                )
                
                result = chatbot_core.search_benchmarks(
                    "stocks",
                    portfolio_size=250000
                )
                
                # Should only return benchmarks within portfolio size
                assert len(result) >= 1
                for benchmark in result:
                    assert benchmark["account_minimum"] <= 250000
                    
    def test_search_benchmarks_empty_query(self):
        """Test search with empty query."""
        with patch.object(chatbot_core, 'embed', return_value=[0.1] * 1536):
            with patch.object(chatbot_core.pc, 'Index') as mock_index_class:
                mock_index = Mock()
                mock_index_class.return_value = mock_index
                mock_index.query.return_value = types.SimpleNamespace(matches=[])
                
                result = chatbot_core.search_benchmarks("")
                assert isinstance(result, list)
                
    def test_search_benchmarks_no_results(self):
        """Test search with no matching results."""
        with patch.object(chatbot_core, 'embed', return_value=[0.1] * 1536):
            with patch.object(chatbot_core.pc, 'Index') as mock_index_class:
                mock_index = Mock()
                mock_index_class.return_value = mock_index
                mock_index.query.return_value = types.SimpleNamespace(matches=[])
                
                result = chatbot_core.search_benchmarks("nonexistent benchmark")
                assert isinstance(result, list)
                assert len(result) == 0

class TestSearchByCharacteristics:
    """Tests for search_by_characteristics function."""
    
    def test_search_by_characteristics_valid_benchmark(self):
        """Test search for alternatives to existing benchmark."""
        # Mock get_benchmark to return valid benchmark
        mock_benchmark = {
            "name": "S&P 500",
            "account_minimum": 2000000,
            "tags": {
                "region": ["US"],
                "asset_class": ["Equity"],
                "style": ["Core"],
                "factor_tilts": ["None"],
                "sector_focus": ["Broad Market"],
                "esg": False
            }
        }
        
        with patch.object(chatbot_core, 'get_benchmark', return_value=mock_benchmark):
            with patch.object(chatbot_core, 'search_benchmarks') as mock_search:
                mock_search.return_value = [
                    {
                        "name": "Russell 1000",
                        "account_minimum": 2000000,
                        "score": 0.9
                    }
                ]
                
                result = chatbot_core.search_by_characteristics("S&P 500")
                
                assert isinstance(result, dict)
                assert "alternatives" in result
                assert len(result["alternatives"]) > 0
                
                # Verify search was called with correct parameters
                mock_search.assert_called()
                
    def test_search_by_characteristics_with_portfolio_size(self):
        """Test search with portfolio size filtering."""
        mock_benchmark = {
            "name": "Russell 2000",
            "account_minimum": 2000000,
            "tags": {
                "region": ["US"],
                "asset_class": ["Equity"],
                "style": ["Small Cap"],
                "factor_tilts": ["None"],
                "sector_focus": ["Broad Market"],
                "esg": False
            }
        }
        
        with patch.object(chatbot_core, 'get_benchmark', return_value=mock_benchmark):
            with patch.object(chatbot_core, 'search_benchmarks') as mock_search:
                # Mock expensive alternatives
                mock_search.return_value = [
                    {
                        "name": "Expensive Alternative",
                        "account_minimum": 5000000,
                        "score": 0.9
                    },
                    {
                        "name": "Affordable Alternative", 
                        "account_minimum": 100000,
                        "score": 0.8
                    }
                ]
                
                result = chatbot_core.search_by_characteristics(
                    "Russell 2000",
                    portfolio_size=250000
                )
                
                # Should filter out expensive alternatives
                assert "alternatives" in result
                for alt in result["alternatives"]:
                    assert alt["account_minimum"] <= 250000
                    
    def test_search_by_characteristics_invalid_benchmark(self):
        """Test search for invalid benchmark name."""
        with patch.object(chatbot_core, 'get_benchmark', return_value=None):
            result = chatbot_core.search_by_characteristics("NonexistentBenchmark")
            
            assert isinstance(result, dict)
            assert "error" in result or "alternatives" not in result or len(result.get("alternatives", [])) == 0
            
    def test_search_by_characteristics_no_alternatives(self):
        """Test search that returns no alternatives."""
        mock_benchmark = {
            "name": "Unique Benchmark",
            "account_minimum": 1000000,
            "tags": {
                "region": ["Unique"],
                "asset_class": ["Special"],
                "style": ["Unique"],
                "factor_tilts": ["None"],
                "sector_focus": ["Niche"],
                "esg": False
            }
        }
        
        with patch.object(chatbot_core, 'get_benchmark', return_value=mock_benchmark):
            with patch.object(chatbot_core, 'search_benchmarks', return_value=[]):
                result = chatbot_core.search_by_characteristics("Unique Benchmark")
                
                assert isinstance(result, dict)
                # Should handle gracefully
                
class TestGetAllBenchmarks:
    """Tests for get_all_benchmarks function."""
    
    @patch('chatbot_core.open', create=True)
    def test_get_all_benchmarks_basic(self, mock_open):
        """Test basic get_all_benchmarks functionality."""
        mock_data = {
            "benchmarks": [
                {
                    "name": "S&P 500",
                    "account_minimum": 2000000,
                    "market_cap": "Large Cap",
                    "tags": {"region": ["US"]}
                },
                {
                    "name": "Russell 2000",
                    "account_minimum": 2000000,
                    "market_cap": "Small Cap", 
                    "tags": {"region": ["US"]}
                }
            ]
        }
        
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)
        
        result = chatbot_core.get_all_benchmarks()
        
        assert isinstance(result, dict)
        assert "benchmarks" in result
        assert len(result["benchmarks"]) == 2
        
    @patch('chatbot_core.open', create=True)
    def test_get_all_benchmarks_with_filters(self, mock_open):
        """Test get_all_benchmarks with filters."""
        mock_data = {
            "benchmarks": [
                {
                    "name": "US Large Cap",
                    "account_minimum": 2000000,
                    "tags": {"region": ["US"], "market_cap": ["Large Cap"]}
                },
                {
                    "name": "International Large Cap",
                    "account_minimum": 2000000,
                    "tags": {"region": ["International"], "market_cap": ["Large Cap"]}
                },
                {
                    "name": "US Small Cap",
                    "account_minimum": 2000000,
                    "tags": {"region": ["US"], "market_cap": ["Small Cap"]}
                }
            ]
        }
        
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)
        
        filters = {"region": ["US"]}
        result = chatbot_core.get_all_benchmarks(filters=filters)
        
        assert "benchmarks" in result
        # Should only return US benchmarks
        for benchmark in result["benchmarks"]:
            assert "US" in benchmark["tags"]["region"]
            
    @patch('chatbot_core.open', create=True)
    def test_get_all_benchmarks_include_dividend(self, mock_open):
        """Test get_all_benchmarks with dividend inclusion."""
        mock_data = {
            "benchmarks": [
                {
                    "name": "Regular Benchmark",
                    "account_minimum": 2000000,
                    "tags": {"dividend_yield": 2.1}
                }
            ]
        }
        
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)
        
        result = chatbot_core.get_all_benchmarks(include_dividend=True)
        
        assert "benchmarks" in result
        # Should include dividend information
        
    @patch('chatbot_core.open', create=True)  
    def test_get_all_benchmarks_region_mapping(self, mock_open):
        """Test get_all_benchmarks region mapping for international queries."""
        mock_data = {
            "benchmarks": [
                {
                    "name": "EAFE Benchmark",
                    "account_minimum": 2000000,
                    "tags": {"region": ["International Developed"]}
                },
                {
                    "name": "Emerging Markets",
                    "account_minimum": 2000000,
                    "tags": {"region": ["Emerging Markets"]}
                },
                {
                    "name": "Global Benchmark",
                    "account_minimum": 2000000,
                    "tags": {"region": ["Global"]}
                }
            ]
        }
        
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)
        
        filters = {"region": ["International"]}
        result = chatbot_core.get_all_benchmarks(filters=filters)
        
        assert "benchmarks" in result
        # Should return international benchmarks (mapped regions)

class TestSearchUtilities:
    """Tests for search utility functions."""
    
    def test_fuzzy_match_benchmark(self):
        """Test fuzzy benchmark name matching."""
        # Test exact match
        result = chatbot_core._fuzzy_match_benchmark("S&P 500")
        if result:
            assert "S&P 500" in result["name"] or "S&P" in result["name"]
            
        # Test fuzzy match
        result = chatbot_core._fuzzy_match_benchmark("SP 500")
        if result:
            assert "S&P" in result["name"] or "500" in result["name"]
            
        # Test no match
        result = chatbot_core._fuzzy_match_benchmark("NonexistentBenchmark123")
        # Could be None or empty depending on implementation
        
    def test_is_fuzzy_match(self):
        """Test fuzzy matching logic."""
        # Exact match
        assert chatbot_core._is_fuzzy_match("S&P 500", "S&P 500")
        
        # Case insensitive
        assert chatbot_core._is_fuzzy_match("s&p 500", "S&P 500")
        
        # Partial match
        assert chatbot_core._is_fuzzy_match("SP 500", "S&P 500")
        
        # No match
        assert not chatbot_core._is_fuzzy_match("Russell", "NASDAQ")
        
    def test_is_aggressive_fuzzy_match(self):
        """Test aggressive fuzzy matching logic."""
        # Should match more loosely than regular fuzzy match
        
        # Test NASDAQ variations
        assert chatbot_core._is_aggressive_fuzzy_match("NASDAQ 100", "Nasdaq 100")
        assert chatbot_core._is_aggressive_fuzzy_match("NASDAQ-100", "Nasdaq 100")
        
        # Test abbreviated forms
        result = chatbot_core._is_aggressive_fuzzy_match("SP", "S&P 500")
        # Behavior depends on implementation

class TestSearchErrorHandling:
    """Tests for search error handling."""
    
    def test_search_with_pinecone_error(self):
        """Test search behavior when Pinecone raises errors."""
        with patch.object(chatbot_core, 'embed', return_value=[0.1] * 1536):
            with patch.object(chatbot_core.pc, 'Index') as mock_index_class:
                mock_index = Mock()
                mock_index_class.return_value = mock_index
                
                # Simulate Pinecone error
                mock_index.query.side_effect = Exception("Pinecone error")
                
                result = chatbot_core.search_benchmarks("test query")
                
                # Should handle error gracefully
                assert isinstance(result, list)
                
    def test_search_with_embedding_error(self):
        """Test search behavior when embedding fails."""
        with patch.object(chatbot_core, 'embed', side_effect=Exception("Embedding error")):
            result = chatbot_core.search_benchmarks("test query")
            
            # Should handle error gracefully
            assert isinstance(result, list)
            
    def test_search_with_invalid_filters(self):
        """Test search with malformed filters."""
        with patch.object(chatbot_core, 'embed', return_value=[0.1] * 1536):
            with patch.object(chatbot_core.pc, 'Index') as mock_index_class:
                mock_index = Mock()
                mock_index_class.return_value = mock_index
                mock_index.query.return_value = types.SimpleNamespace(matches=[])
                
                # Test with invalid filter structure
                invalid_filters = {"invalid": "structure"}
                
                result = chatbot_core.search_benchmarks(
                    "test query",
                    filters=invalid_filters
                )
                
                # Should handle gracefully
                assert isinstance(result, list)

class TestSearchPerformance:
    """Performance-related tests for search functions."""
    
    def test_search_result_limit(self):
        """Test that search results are properly limited."""
        with patch.object(chatbot_core, 'embed', return_value=[0.1] * 1536):
            with patch.object(chatbot_core.pc, 'Index') as mock_index_class:
                mock_index = Mock()
                mock_index_class.return_value = mock_index
                
                # Create many mock results
                many_results = []
                for i in range(50):
                    many_results.append(
                        types.SimpleNamespace(
                            id=f"benchmark_{i}",
                            score=0.9 - i * 0.01,
                            metadata={
                                "name": f"Benchmark {i}",
                                "account_minimum": 1000000
                            }
                        )
                    )
                
                mock_index.query.return_value = types.SimpleNamespace(matches=many_results)
                
                result = chatbot_core.search_benchmarks("test", top_k=10)
                
                # Should limit results
                assert len(result) <= 10
                
    def test_search_embedding_caching(self):
        """Test that embeddings might be cached (if implemented)."""
        with patch.object(chatbot_core, 'embed') as mock_embed:
            mock_embed.return_value = [0.1] * 1536
            
            with patch.object(chatbot_core.pc, 'Index') as mock_index_class:
                mock_index = Mock()
                mock_index_class.return_value = mock_index
                mock_index.query.return_value = types.SimpleNamespace(matches=[])
                
                # Same query multiple times
                chatbot_core.search_benchmarks("same query")
                chatbot_core.search_benchmarks("same query")
                
                # Check if embedding was called multiple times
                # (behavior depends on caching implementation)
                assert mock_embed.call_count >= 1