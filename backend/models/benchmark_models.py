"""Data models for benchmark data structures."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class BenchmarkSearchResult:
    """Represents a benchmark search result with relevance scoring."""
    benchmark: Dict[str, Any]
    relevance_score: float
    match_reason: Optional[str] = None
    
    def __post_init__(self):
        """Ensure benchmark has required fields."""
        if not isinstance(self.benchmark, dict):
            raise ValueError("Benchmark must be a dictionary")
        if 'name' not in self.benchmark:
            raise ValueError("Benchmark must have a 'name' field")


@dataclass  
class MinimumResult:
    """Represents the result of a minimum investment query."""
    name: str
    account_minimum: str
    account_minimum_value: int
    dividend_yield: Optional[str] = None
    dividend_yield_value: Optional[float] = None
    found: bool = True
    message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API responses."""
        result = {
            'name': self.name,
            'account_minimum': self.account_minimum,
            'account_minimum_value': self.account_minimum_value,
            'found': self.found
        }
        
        if self.dividend_yield is not None:
            result['dividend_yield'] = self.dividend_yield
            result['dividend_yield_value'] = self.dividend_yield_value
            
        if self.message is not None:
            result['message'] = self.message
            
        return result


@dataclass
class BlendResult:
    """Represents the result of a blended minimum calculation."""
    total_minimum: str
    total_minimum_value: int
    breakdown: List[Dict[str, Any]]
    weighted_dividend_yield: Optional[str] = None
    weighted_dividend_yield_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API responses."""
        result = {
            'total_minimum': self.total_minimum,
            'total_minimum_value': self.total_minimum_value,
            'breakdown': self.breakdown
        }
        
        if self.weighted_dividend_yield is not None:
            result['weighted_dividend_yield'] = self.weighted_dividend_yield
            result['weighted_dividend_yield_value'] = self.weighted_dividend_yield_value
            
        return result


@dataclass
class SearchFilters:
    """Represents search filters for benchmark queries."""
    region: Optional[str] = None
    asset_class: Optional[str] = None
    style: Optional[str] = None
    sector: Optional[str] = None
    max_minimum: Optional[int] = None
    min_yield: Optional[float] = None
    max_yield: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchFilters':
        """Create SearchFilters from dictionary data."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})