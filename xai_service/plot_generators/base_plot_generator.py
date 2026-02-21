"""
Base Plot Generator - Abstract base class for all plot generators
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd


class BasePlotGenerator(ABC):
    """Abstract base class for all plot generators"""
    
    def __init__(self, data_mode: str):
        self.data_mode = data_mode
        self.supported_plot_types = []
    
    @abstractmethod
    def generate_plot(self, query: str, data: pd.DataFrame, 
                     user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate interactive plot from natural language query.
        
        Args:
            query: Natural language query from user
            data: DataFrame with user's data
            user_id: User identifier
            context: Additional context (model info, metadata, etc.)
        
        Returns:
            Dict with 'plot_html', 'plot_type', 'metadata'
        """
        pass
    
    @abstractmethod
    def get_supported_queries(self) -> list:
        """Return examples of supported query types"""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate if data is suitable for this generator
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

