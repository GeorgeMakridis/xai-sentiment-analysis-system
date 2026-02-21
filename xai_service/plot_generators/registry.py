"""
Plot Generator Registry - Manages plot generators for different data modes
"""

from typing import Dict, Type
from .base_plot_generator import BasePlotGenerator
from .sentiment_plot_generator import SentimentPlotGenerator
from .image_plot_generator import ImagePlotGenerator

class PlotGeneratorRegistry:
    """Registry for plot generators by data mode"""
    
    _generators: Dict[str, Type[BasePlotGenerator]] = {
        'sentiment': SentimentPlotGenerator,
        'text': SentimentPlotGenerator,  # Alias for sentiment
        'image': ImagePlotGenerator,
    }
    
    @classmethod
    def get_generator(cls, data_mode: str) -> BasePlotGenerator:
        """Get plot generator for data mode"""
        if data_mode not in cls._generators:
            raise ValueError(f"Unsupported data mode: {data_mode}")
        
        generator_class = cls._generators[data_mode]
        return generator_class()
    
    @classmethod
    def register_generator(cls, data_mode: str, generator_class: Type[BasePlotGenerator]):
        """Register a new plot generator (for future extensibility)"""
        cls._generators[data_mode] = generator_class
    
    @classmethod
    def list_supported_modes(cls) -> list:
        """List all supported data modes"""
        return list(cls._generators.keys())

