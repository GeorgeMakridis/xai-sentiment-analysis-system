"""
Plot Generators Module

Modular system for generating interactive plots from natural language queries.
Supports multiple data modes (sentiment, timeseries, tabular, image, etc.)
"""

from .base_plot_generator import BasePlotGenerator
from .sentiment_plot_generator import SentimentPlotGenerator
from .image_plot_generator import ImagePlotGenerator
from .registry import PlotGeneratorRegistry

__all__ = [
    'BasePlotGenerator',
    'SentimentPlotGenerator',
    'ImagePlotGenerator',
    'PlotGeneratorRegistry'
]

