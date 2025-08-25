"""Benchmark service for data access, fuzzy matching, and minimum calculations."""

import json
import logging
from typing import List, Dict, Any, Optional
import re

from services.base_service import BaseService
from models.benchmark_models import MinimumResult, BlendResult
from utils.description_utils import build_semantic_description

logger = logging.getLogger(__name__)


class BenchmarkService(BaseService):
    """Service for benchmark data operations and minimum calculations."""
    
    def __init__(self):
        super().__init__()
        self.validate_config()
        self._benchmark_map = None
        self._load_benchmark_map()
    
    def _load_benchmark_map(self):
        """Load and cache benchmark mapping."""
        if self._benchmark_map is None:
            try:
                benchmarks = self.load_benchmarks_data()
                
                # Ensure all benchmarks have descriptions and account_minimum_value fields
                for bench in benchmarks:
                    if "description" not in bench:
                        bench["description"] = build_semantic_description(bench)
                    
                    # Add account_minimum_value field for portfolio filtering
                    if "account_minimum_value" not in bench and "account_minimum" in bench:
                        bench["account_minimum_value"] = bench["account_minimum"]
                
                self._benchmark_map = {bench["name"].lower(): bench for bench in benchmarks}
                logger.info(f"Loaded {len(benchmarks)} benchmarks successfully")
                
            except Exception as e:
                logger.error(f"Failed to load benchmark data: {e}")
                raise
    
    def get_benchmark(self, name: str) -> Optional[Dict[str, Any]]:
        """Return benchmark data by name using exact match or fuzzy matching."""
        if not name:
            return None
            
        # First try exact match
        exact_match = self._benchmark_map.get(name.lower())
        if exact_match:
            return exact_match
        
        # If no exact match, try fuzzy matching
        return self._fuzzy_match_benchmark(name)
    
    def _fuzzy_match_benchmark(self, name: str) -> Optional[Dict[str, Any]]:
        """Find benchmark using fuzzy matching for common variations."""
        name_lower = name.lower().strip()
        
        # Common abbreviation and variation mappings
        fuzzy_mappings = {
            # US/USA variations
            'msci world ex us': 'msci world ex usa',
            'msci world ex-us': 'msci world ex usa', 
            'msci world excluding us': 'msci world ex usa',
            'msci world excluding usa': 'msci world ex usa',
            
            # NASDAQ variations - map all variants to exact key
            'nasdaq-100': 'nasdaq 100',
            'nasdaq 100 index': 'nasdaq 100',
            'nasdaq-100 index': 'nasdaq 100',
            'nasdaq100': 'nasdaq 100',
            
            # S&P variations
            's and p 500': 's&p 500',
            's & p 500': 's&p 500',
            'sp 500': 's&p 500',
            's&p500': 's&p 500',
            
            # Russell variations
            'russell 2000 value index': 'russell 2000 value',
            'russell 1000 value index': 'russell 1000 value',
            'russell 2000 growth index': 'russell 2000 growth',
            'russell 1000 growth index': 'russell 1000 growth',
            
            # MSCI variations
            'msci eafe index': 'msci eafe',
            'msci world index': 'msci world',
            'msci acwi index': 'msci acwi',
            'msci emerging markets index': 'msci emerging markets',
            
            # ESG variations
            'msci world esg': 'msci world esg leaders',
            'msci acwi esg': 'msci acwi esg universal',
        }
        
        # Try direct fuzzy mappings first
        if name_lower in fuzzy_mappings:
            mapped_name = fuzzy_mappings[name_lower]
            result = self._benchmark_map.get(mapped_name)
            if result:
                logger.info(f"Fuzzy match: '{name}' -> '{result['name']}'")
                return result
        
        # Try partial matching with benchmark names
        for benchmark_key, benchmark_data in self._benchmark_map.items():
            # Calculate similarity score
            if self._is_fuzzy_match(name_lower, benchmark_key):
                logger.info(f"Fuzzy match: '{name}' -> '{benchmark_data['name']}'")
                return benchmark_data
        
        # If still no match, try more aggressive partial matching
        for benchmark_key, benchmark_data in self._benchmark_map.items():
            if self._is_aggressive_fuzzy_match(name_lower, benchmark_key):
                logger.info(f"Aggressive fuzzy match: '{name}' -> '{benchmark_data['name']}'")
                return benchmark_data
        
        return None
    
    def _is_fuzzy_match(self, input_name: str, benchmark_key: str) -> bool:
        """Check if input name is a fuzzy match for benchmark key."""
        # Remove common words and punctuation for comparison
        def clean_name(name):
            # Remove common words that don't affect identity
            common_words = ['index', 'fund', 'etf', 'the', 'of', 'and', '&']
            words = name.lower().replace('-', ' ').replace('_', ' ').split()
            return ' '.join([w for w in words if w not in common_words])
        
        cleaned_input = clean_name(input_name)
        cleaned_benchmark = clean_name(benchmark_key)
        
        # Check if input is contained in benchmark name (with word boundaries)
        if cleaned_input in cleaned_benchmark:
            return True
        
        # Check if benchmark name is contained in input (for longer inputs)
        if cleaned_benchmark in cleaned_input:
            return True
        
        # Check word overlap - at least 70% of words should match
        input_words = set(cleaned_input.split())
        benchmark_words = set(cleaned_benchmark.split())
        
        if len(input_words) == 0 or len(benchmark_words) == 0:
            return False
        
        overlap = len(input_words & benchmark_words)
        overlap_ratio = overlap / min(len(input_words), len(benchmark_words))
        
        return overlap_ratio >= 0.7
    
    def _is_aggressive_fuzzy_match(self, input_name: str, benchmark_key: str) -> bool:
        """More aggressive fuzzy matching for partial names."""
        # Extract key identifying terms
        input_terms = set(re.findall(r'\b\w+\b', input_name.lower()))
        benchmark_terms = set(re.findall(r'\b\w+\b', benchmark_key.lower()))
        
        # Remove common stop words
        stop_words = {'index', 'fund', 'etf', 'the', 'of', 'and', 'a', 'an'}
        input_terms -= stop_words
        benchmark_terms -= stop_words
        
        if not input_terms:
            return False
        
        # For very short inputs (1-2 words), require high overlap
        if len(input_terms) <= 2:
            return len(input_terms & benchmark_terms) >= len(input_terms)
        
        # For longer inputs, require at least 60% overlap
        overlap_ratio = len(input_terms & benchmark_terms) / len(input_terms)
        return overlap_ratio >= 0.6
    
    def get_minimum(self, name: str, include_dividend: bool = False) -> MinimumResult:
        """Get minimum investment amount for a benchmark."""
        try:
            bench = self.get_benchmark(name)
            if bench:
                result = MinimumResult(
                    name=bench["name"],
                    account_minimum=bench["account_minimum"],
                    account_minimum_value=bench.get("account_minimum_value", 0)
                )
                
                if include_dividend:
                    dividend_data = bench.get("fundamentals", {})
                    result.dividend_yield = dividend_data.get("dividend_yield_display")
                    result.dividend_yield_value = dividend_data.get("dividend_yield")
                
                return result
            else:
                return MinimumResult(
                    name=name,
                    account_minimum="N/A",
                    account_minimum_value=0,
                    found=False,
                    message=f"Benchmark '{name}' not found"
                )
                
        except Exception as e:
            logger.error(f"Get minimum failed for '{name}': {e}")
            return MinimumResult(
                name=name,
                account_minimum="N/A",
                account_minimum_value=0,
                found=False,
                message=f"Unable to retrieve minimum for '{name}' due to system error"
            )
    
    def blend_minimum(self, allocations: List[Dict[str, Any]], include_dividend: bool = False) -> BlendResult:
        """Calculate blended minimum investment amount."""
        try:
            if not allocations:
                raise ValueError("No allocations provided")
                
            total_weight = sum(a.get("weight", 0) for a in allocations)
            if abs(total_weight - 1.0) > 1e-6:
                raise ValueError(f"Weights sum to {total_weight:.3f}, must sum to 1.0")
                
            total = 0.0
            weighted_yield = 0.0
            has_yield = True
            breakdown = []
            
            for allocation in allocations:
                bench = self.get_benchmark(allocation.get("name", ""))
                if not bench:
                    raise ValueError(f"Benchmark '{allocation.get('name')}' not found")
                    
                weight = allocation["weight"]
                minimum_value = bench.get("account_minimum_value", 0)
                weighted_minimum = minimum_value * weight
                total += weighted_minimum
                
                breakdown.append({
                    "name": bench["name"],
                    "weight": weight,
                    "account_minimum": bench["account_minimum"],
                    "weighted_minimum": f"${weighted_minimum:,.0f}"
                })
                
                # Handle dividend yield
                if include_dividend:
                    dividend_data = bench.get("fundamentals", {})
                    dy = dividend_data.get("dividend_yield")
                    if dy is None:
                        has_yield = False
                    else:
                        weighted_yield += dy * weight
            
            result = BlendResult(
                total_minimum=f"${total:,.0f}",
                total_minimum_value=int(total),
                breakdown=breakdown
            )
            
            if include_dividend and has_yield:
                result.weighted_dividend_yield = f"{weighted_yield:.2f}%"
                result.weighted_dividend_yield_value = weighted_yield
                
            return result
            
        except Exception as e:
            logger.error(f"Blend minimum calculation failed: {e}")
            raise
    
    def get_all_benchmarks(
        self, 
        include_dividend: bool = False, 
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get all benchmarks with optional filtering."""
        try:
            benchmarks = self.load_benchmarks_data()
            
            if filters:
                filtered_benchmarks = []
                for bench in benchmarks:
                    if self._matches_filters(bench, filters):
                        filtered_benchmarks.append(bench)
                benchmarks = filtered_benchmarks
            
            # Transform to response format
            results = []
            for bench in benchmarks:
                item = {
                    "name": bench["name"],
                    "account_minimum": bench["account_minimum"],
                    "description": bench.get("description", "")
                }
                if include_dividend:
                    dividend_data = bench.get("fundamentals", {})
                    item["dividend_yield"] = dividend_data.get("dividend_yield_display")
                results.append(item)
            
            # Handle region mapping for international queries
            if filters and 'region' in filters:
                region_query = filters['region'].lower() if isinstance(filters['region'], str) else ''
                if 'international' in region_query:
                    # Map "international" to actual region values in the data
                    region_mappings = {
                        "international": ["International Developed", "Global", "Emerging Markets"]
                    }
                    
                    expanded_regions = []
                    for region in region_mappings.get("international", []):
                        expanded_regions.append(region)
                    
                    # Re-filter with expanded regions
                    international_benchmarks = []
                    for bench in self.load_benchmarks_data():
                        bench_regions = bench.get("tags", {}).get("region", [])
                        if any(region in bench_regions for region in expanded_regions):
                            item = {
                                "name": bench["name"],
                                "account_minimum": bench["account_minimum"],
                                "description": bench.get("description", "")
                            }
                            if include_dividend:
                                dividend_data = bench.get("fundamentals", {})
                                item["dividend_yield"] = dividend_data.get("dividend_yield_display")
                            international_benchmarks.append(item)
                    
                    results = international_benchmarks
            
            return {
                "benchmarks": results,
                "count": len(results),
                "message": f"Found {len(results)} benchmark(s)" + (f" matching filters" if filters else "")
            }
            
        except Exception as e:
            logger.error(f"Get all benchmarks failed: {e}")
            return {
                "benchmarks": [],
                "count": 0,
                "error": f"Unable to retrieve benchmarks: {str(e)}"
            }
    
    def _matches_filters(self, benchmark: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if benchmark matches the provided filters."""
        tags = benchmark.get("tags", {})
        fundamentals = benchmark.get("fundamentals", {})
        
        for filter_key, filter_value in filters.items():
            if filter_key == "region":
                benchmark_regions = tags.get("region", [])
                if isinstance(filter_value, list):
                    if not any(region in benchmark_regions for region in filter_value):
                        return False
                else:
                    if filter_value not in benchmark_regions:
                        return False
                        
            elif filter_key == "asset_class":
                benchmark_classes = tags.get("asset_class", [])
                if isinstance(filter_value, list):
                    if not any(cls in benchmark_classes for cls in filter_value):
                        return False
                else:
                    if filter_value not in benchmark_classes:
                        return False
                        
            elif filter_key == "style":
                benchmark_styles = tags.get("style", [])
                if isinstance(filter_value, list):
                    if not any(style in benchmark_styles for style in filter_value):
                        return False
                else:
                    if filter_value not in benchmark_styles:
                        return False
                        
            elif filter_key == "max_minimum":
                min_value = benchmark.get("account_minimum_value", 0)
                if min_value > filter_value:
                    return False
                    
            elif filter_key == "min_yield":
                yield_value = fundamentals.get("dividend_yield", 0)
                if yield_value < filter_value:
                    return False
                    
            elif filter_key == "max_yield":
                yield_value = fundamentals.get("dividend_yield", float('inf'))
                if yield_value > filter_value:
                    return False
        
        return True
    
    def clean_cache(self):
        """Clear all caches."""
        super().clean_cache()
        self._benchmark_map = None