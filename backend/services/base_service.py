"""Base service class with common patterns."""

import json
import logging
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class BaseService:
    """Base service with common functionality."""
    
    def __init__(self):
        self._benchmarks_cache = None
        
    def load_benchmarks_data(self) -> List[Dict[str, Any]]:
        """Load benchmarks data with caching."""
        if self._benchmarks_cache is None:
            try:
                with open("config/benchmarks.json", "r") as f:
                    data = json.load(f)
                    self._benchmarks_cache = data.get("benchmarks", [])
            except Exception as e:
                logger.error(f"Could not load benchmarks data: {e}")
                self._benchmarks_cache = []
        return self._benchmarks_cache
    
    def validate_config(self):
        """Validate service configuration."""
        # Validate required environment variables
        required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENV"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        # Validate config files exist
        config_files = ["config/benchmarks.json", "config/system_prompt.txt"]
        missing_files = [f for f in config_files if not os.path.exists(f)]
        
        if missing_files:
            raise FileNotFoundError(f"Missing required config files: {', '.join(missing_files)}")
            
    def clean_cache(self):
        """Clear internal caches."""
        self._benchmarks_cache = None