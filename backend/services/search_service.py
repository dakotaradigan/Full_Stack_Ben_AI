"""Search service for Pinecone vector operations and benchmark search."""

import json
import logging
import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from openai import OpenAI

from services.base_service import BaseService
from models.chat_models import CircuitBreaker
from models.benchmark_models import SearchFilters

logger = logging.getLogger(__name__)


class SearchService(BaseService):
    """Service for vector search operations using Pinecone."""
    
    def __init__(self):
        super().__init__()
        self.validate_config()
        
        # Initialize clients
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Get the index
        self.index = self.pinecone_client.Index("benchmark-index")
        
        # Configuration
        self.embedding_model = "text-embedding-3-small"
        
        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    
    def embed(self, text: str) -> List[float]:
        """Create embeddings with error handling and circuit breaker."""
        try:
            resp = self.circuit_breaker.call(
                self.openai_client.embeddings.create,
                model=self.embedding_model,
                input=text[:8000]  # Truncate to avoid token limits
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise
    
    def search_benchmarks(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_dividend: bool = False,
        portfolio_size: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search benchmarks with enhanced error handling, sector-specific boosting, and portfolio size filtering."""
        try:
            if not query.strip():
                return []
                
            # Preprocess query for sector-specific searches
            query_lower = query.lower().strip()
            
            vec = self.embed(query)
            
            pinecone_filter: Optional[Dict[str, Any]] = None
            if filters:
                pinecone_filter = {}
                for key, value in filters.items():
                    if isinstance(value, dict) and any(k.startswith("$") for k in value):
                        pinecone_filter[key] = value
                    else:
                        pinecone_filter[key] = {"$eq": value}

            res = self.index.query(
                vector=vec,
                top_k=min(top_k, 20),  # Cap at 20 results
                include_metadata=True,
                **({"filter": pinecone_filter} if pinecone_filter else {}),
            )
            
            results = []
            for match in res.matches:
                bench = match.metadata
                item = {
                    "name": bench["name"],
                    "account_minimum": bench["account_minimum"],
                    "description": bench.get("description"),
                    "score": match.score,
                }
                if include_dividend:
                    item["dividend_yield"] = bench.get("dividend_yield")
                results.append(item)
            
            # Apply sector boosting and fallback injection
            results = self._apply_sector_boosting(results, query_lower)
            results = self._inject_missing_sector_benchmarks(results, query_lower)
            results = self._prioritize_standard_benchmarks(results, query_lower)
            
            # Apply portfolio size filtering if specified
            if portfolio_size is not None:
                filtered_results = []
                for result in results:
                    min_value = result.get("account_minimum", 0)
                    if isinstance(min_value, (int, float)) and min_value <= portfolio_size:
                        filtered_results.append(result)
                        logger.info(f"PORTFOLIO FILTER: Including {result['name']} (${min_value:,}) for ${portfolio_size:,} portfolio")
                    elif isinstance(min_value, (int, float)):
                        logger.info(f"PORTFOLIO FILTER: Excluding {result['name']} (${min_value:,}) for ${portfolio_size:,} portfolio")
                results = filtered_results
            
            logger.info(f"Search query '{query[:50]}...' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search benchmarks failed: {e}")
            return []
    
    def search_by_characteristics(
        self,
        reference_benchmark: str,
        portfolio_size: Optional[float] = None,
        top_k: int = 5,
        include_dividend: bool = False,
        get_benchmark_func=None  # Injected dependency
    ) -> List[Dict[str, Any]]:
        """Search by characteristics with enhanced error handling."""
        try:
            logger.info(f"Search by characteristics: '{reference_benchmark}', portfolio_size=${portfolio_size:,}" if portfolio_size else f"Search by characteristics: '{reference_benchmark}', no portfolio_size")
            
            if not get_benchmark_func:
                raise ValueError("get_benchmark_func is required for search_by_characteristics")
                
            ref_bench = get_benchmark_func(reference_benchmark)
            if not ref_bench:
                logger.warning(f"Reference benchmark '{reference_benchmark}' not found")
                return []
            
            ref_tags = ref_bench.get("tags", {})
            
            # Build filters based on reference benchmark characteristics
            filters = {}
            
            # Core characteristics that should match (in order of importance)
            if ref_tags.get("region"):
                filters["region"] = {"$in": ref_tags["region"]}
            if ref_tags.get("asset_class"):
                filters["asset_class"] = {"$in": ref_tags["asset_class"]}
            if ref_tags.get("style"):
                filters["style"] = {"$in": ref_tags["style"]}
            
            # ESG matching is important if specified
            if ref_tags.get("esg") is not None:
                filters["esg"] = {"$eq": ref_tags["esg"]}
            
            # Create search query
            query_parts = []
            for key in ["region", "style", "factor_tilts", "sector_focus"]:
                if ref_tags.get(key):
                    query_parts.extend(ref_tags[key])
            
            query = " ".join(query_parts) if query_parts else reference_benchmark
            
            # Search with characteristics-based filters
            if portfolio_size is not None:
                results = self._search_viable_alternatives(
                    query=query,
                    portfolio_size=portfolio_size,
                    top_k=top_k,
                    filters=filters,
                    include_dividend=include_dividend,
                    get_benchmark_func=get_benchmark_func
                )
            else:
                results = self.search_benchmarks(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    include_dividend=include_dividend
                )
            
            # Filter out the reference benchmark itself
            results = [r for r in results if r["name"] != reference_benchmark]
            
            # Multi-level fallback for geographic consistency
            if not results:
                results = self._fallback_search_by_region(
                    ref_bench, reference_benchmark, query, portfolio_size, 
                    include_dividend, get_benchmark_func
                )
            
            logger.info(f"Search by characteristics for '{reference_benchmark}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search by characteristics failed: {e}")
            return []
    
    def _apply_sector_boosting(self, results: List[Dict[str, Any]], query_lower: str) -> List[Dict[str, Any]]:
        """Boost sector-specific benchmarks for relevant queries."""
        boosted_results = []
        for item in results:
            name_lower = item["name"].lower()
            description_lower = item.get("description", "").lower()
            boosted_score = item["score"]
            
            # Technology sector boosting - prioritize NASDAQ-100 for tech searches
            if any(keyword in query_lower for keyword in ["technology", "tech"]):
                if "nasdaq 100" in name_lower or "nasdaq-100" in name_lower:
                    boosted_score *= 2.0  # Double the relevance score
                    logger.info(f"SECTOR BOOST: Boosted {item['name']} for technology query")
                elif any(tech_word in description_lower for tech_word in ["technology", "tech"]):
                    boosted_score *= 1.5  # Moderate boost for tech-related benchmarks
            
            # International/Global boosting
            elif any(keyword in query_lower for keyword in ["international", "global", "world"]):
                if any(global_word in name_lower for global_word in ["world", "acwi", "eafe", "international", "global"]):
                    boosted_score *= 1.5
                    logger.info(f"SECTOR BOOST: Boosted {item['name']} for global/international query")
            
            # ESG boosting
            elif any(keyword in query_lower for keyword in ["esg", "sustainable", "clean"]):
                if any(esg_word in name_lower for esg_word in ["esg", "clean", "sustainable"]):
                    boosted_score *= 1.5
                    logger.info(f"SECTOR BOOST: Boosted {item['name']} for ESG query")
            
            item_copy = item.copy()
            item_copy["score"] = boosted_score
            boosted_results.append(item_copy)
        
        return boosted_results
    
    def _inject_missing_sector_benchmarks(self, results: List[Dict[str, Any]], query_lower: str) -> List[Dict[str, Any]]:
        """SCALABLE: Ensure key sector benchmarks appear using data-driven detection."""
        benchmarks = self.load_benchmarks_data()
        
        # Data-driven sector detection using benchmark tags
        sector_matches = []
        
        # Check each benchmark for semantic matches to query
        for benchmark in benchmarks:
            name = benchmark.get("name", "").lower()
            description = benchmark.get("description", "").lower()
            tags = benchmark.get("tags", {})
            
            # Factor detection
            if any(factor_word in query_lower for factor_word in ["factor", "value", "growth", "momentum", "quality", "dividend"]):
                factor_tilts = tags.get("factor_tilts", [])
                if factor_tilts or "value" in description or "growth" in description or "dividend" in description:
                    sector_matches.append(benchmark)
            
            # Technology detection  
            elif any(tech_word in query_lower for tech_word in ["technology", "tech", "nasdaq"]):
                sector_focus = tags.get("sector_focus", [])
                if "technology" in name or any("technology" in sf.lower() for sf in sector_focus):
                    sector_matches.append(benchmark)
            
            # International/Global detection
            elif any(global_word in query_lower for global_word in ["international", "global", "world"]):
                regions = tags.get("region", [])
                if any(region.lower() in ["global", "international developed", "world"] for region in regions):
                    sector_matches.append(benchmark)
            
            # Small/Large cap detection
            elif any(size_word in query_lower for size_word in ["small", "large", "mid"]):
                styles = tags.get("style", [])
                for style in styles:
                    if any(size_word.title() in style for size_word in ["small", "large", "mid"] if size_word in query_lower):
                        sector_matches.append(benchmark)
                        break
        
        # Convert sector matches to search result format and inject missing ones
        result_names = [r["name"].lower() for r in results]
        injected_results = []
        
        for match in sector_matches[:3]:  # Limit to top 3 sector matches
            if match["name"].lower() not in result_names:
                injected_item = {
                    "name": match["name"],
                    "account_minimum": match["account_minimum"],
                    "description": match.get("description"),
                    "score": 0.8,  # High score to ensure it appears near the top
                }
                injected_results.append(injected_item)
                logger.info(f"SECTOR FALLBACK: Injected {match['name']} for data-driven sector match")
        
        return injected_results + results
    
    def _prioritize_standard_benchmarks(self, results: List[Dict[str, Any]], query_lower: str) -> List[Dict[str, Any]]:
        """Sort results to prioritize standard benchmarks over specialized ones."""
        def sort_key(item):
            name = item["name"].lower()
            
            # Priority 1: Standard major benchmarks
            standard_indicators = ["msci acwi", "msci eafe", "msci world", "s&p 500", "russell 2000", "russell 1000", "nasdaq 100"]
            if any(indicator in name for indicator in standard_indicators):
                # Check if it's NOT an ESG/specialized version (unless we're searching for ESG)
                if not any(special in name for special in ["esg", "clean", "sustainable", "factor", "momentum"]) or any(keyword in query_lower for keyword in ["esg", "sustainable", "clean"]):
                    return (0, -item["score"])  # Highest priority, then by score
            
            # Priority 2: Other standard benchmarks
            if not any(special in name for special in ["esg", "clean", "sustainable", "factor", "momentum"]) or any(keyword in query_lower for keyword in ["esg", "sustainable", "clean"]):
                return (1, -item["score"])
            
            # Priority 3: Specialized/ESG benchmarks
            return (2, -item["score"])
        
        return sorted(results, key=sort_key)
    
    def _search_viable_alternatives(
        self,
        query: str,
        portfolio_size: float,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_dividend: bool = False,
        get_benchmark_func=None
    ) -> List[Dict[str, Any]]:
        """Search for viable alternatives based on portfolio size constraints."""
        # Get broader search results first
        broader_results = self.search_benchmarks(
            query=query,
            top_k=top_k * 3,  # Get more results to filter
            filters=filters,
            include_dividend=include_dividend
        )
        
        viable_results = []
        for result in broader_results:
            # Parse minimum investment amount
            min_value_raw = result.get("account_minimum", "N/A")
            if min_value_raw == "N/A":
                continue
                
            try:
                # Handle both numeric and string minimum values
                if isinstance(min_value_raw, (int, float)):
                    min_value = float(min_value_raw)
                else:
                    # Extract numeric value from string format
                    min_value = self._parse_minimum_amount(str(min_value_raw))
                    
                if min_value <= portfolio_size:
                    viable_results.append(result)
            except:
                continue  # Skip if we can't parse the minimum
        
        # Defensive portfolio size filtering - ensure NO expensive alternatives slip through
        final_results = []
        for result in viable_results:
            if get_benchmark_func:
                bench = get_benchmark_func(result["name"])
                if bench:
                    account_minimum_value = bench.get("account_minimum_value", 0)
                    if account_minimum_value <= portfolio_size:
                        final_results.append(result)
                        logger.info(f"PORTFOLIO FILTER: {result['name']} (${account_minimum_value:,}) included for portfolio ${portfolio_size:,}")
                    else:
                        logger.info(f"PORTFOLIO FILTER: {result['name']} (${account_minimum_value:,}) excluded for portfolio ${portfolio_size:,}")
                else:
                    final_results.append(result)  # Include if we can't verify
            else:
                final_results.append(result)
        
        return final_results[:top_k]
    
    def _parse_minimum_amount(self, min_str: str) -> float:
        """Parse minimum amount string to numeric value."""
        import re
        
        if not min_str or min_str == "N/A":
            return 0
        
        # Remove currency symbols and commas
        clean_str = re.sub(r'[$,]', '', min_str)
        
        # Handle K and M suffixes
        if 'K' in clean_str.upper():
            return float(re.sub(r'[^\d.]', '', clean_str)) * 1000
        elif 'M' in clean_str.upper():
            return float(re.sub(r'[^\d.]', '', clean_str)) * 1000000
        else:
            return float(re.sub(r'[^\d.]', '', clean_str))
    
    def _fallback_search_by_region(
        self, 
        ref_bench: Dict[str, Any], 
        reference_benchmark: str, 
        query: str,
        portfolio_size: Optional[float],
        include_dividend: bool,
        get_benchmark_func
    ) -> List[Dict[str, Any]]:
        """Fallback search prioritizing geographic consistency."""
        ref_tags = ref_bench.get("tags", {})
        fallback_results = []
        
        # PRIORITY 1: Region match is CRITICAL for geographic consistency
        if ref_tags.get("region"):
            region_filters = {"region": {"$in": ref_tags["region"]}}
            region_results = self._search_with_filters_helper(
                query, region_filters, portfolio_size, 5, include_dividend, get_benchmark_func
            )
            region_results = [r for r in region_results if r["name"] != reference_benchmark]
            
            # For Global benchmarks, prefer other Global or International alternatives
            ref_region = ref_tags.get("region", [])
            is_global_reference = any(region.lower() in ["global", "international developed", "world"] for region in ref_region)
            
            if is_global_reference and region_results:
                # Prioritize global/international alternatives over regional ones
                global_results = []
                other_results = []
                
                for r in region_results:
                    if get_benchmark_func:
                        r_bench = get_benchmark_func(r["name"])
                        r_tags = r_bench.get("tags", {}) if r_bench else {}
                        r_regions = r_tags.get("region", [])
                        is_global = any(region.lower() in ["global", "international developed", "world"] for region in r_regions)
                        
                        if is_global:
                            global_results.append(r)
                        else:
                            other_results.append(r)
                    else:
                        other_results.append(r)
                
                # Prefer global results, then international
                preferred_results = global_results + other_results
                if preferred_results:
                    fallback_results.extend(preferred_results[:3])
                    logger.info(f"REGION CONSISTENCY: Found {len(preferred_results)} region-consistent alternatives for global benchmark")
            elif region_results:
                fallback_results.extend(region_results[:3])
        
        return fallback_results
    
    def _search_with_filters_helper(
        self, 
        query: str, 
        filters: Dict[str, Any], 
        portfolio_size: Optional[float], 
        top_k: int, 
        include_dividend: bool,
        get_benchmark_func
    ) -> List[Dict[str, Any]]:
        """Helper method for filtered search."""
        if portfolio_size is not None:
            return self._search_viable_alternatives(
                query=query,
                portfolio_size=portfolio_size,
                top_k=top_k,
                filters=filters,
                include_dividend=include_dividend,
                get_benchmark_func=get_benchmark_func
            )
        else:
            return self.search_benchmarks(
                query=query,
                top_k=top_k,
                filters=filters,
                include_dividend=include_dividend
            )