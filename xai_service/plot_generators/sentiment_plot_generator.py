"""
Sentiment Plot Generator - Generates interactive plots for sentiment analysis data
"""

from .base_plot_generator import BasePlotGenerator
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)

# Try to import openai, but allow fallback if not available
try:
    import openai
    OPENAI_AVAILABLE = True
    # Configure OpenAI if API key is available
    import os
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key and api_key != 'your-openai-api-key':
        openai.api_key = api_key
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
except Exception as e:
    OPENAI_AVAILABLE = False
    openai = None
    logger.warning(f"OpenAI configuration failed: {e}")


class SentimentPlotGenerator(BasePlotGenerator):
    """Plot generator for sentiment analysis data"""
    
    def __init__(self):
        super().__init__('sentiment')
        self.supported_plot_types = [
            'sentiment_over_time',
            'sentiment_by_asset',
            'word_sentiment_association',
            'sentiment_distribution',
            'sentiment_polarity_distribution',
            'sentiment_by_period',
            'keywords_by_sentiment',
            'bar',
            'line',
            'scatter',
            'heatmap',
            'box',
            'histogram'
        ]
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Check if data has sentiment-related columns"""
        # Check for sentiment column
        sentiment_cols = [col for col in data.columns if 'sentiment' in col.lower()]
        if not sentiment_cols:
            return False, "Data missing 'sentiment' column"
        
        # Check if we have at least some data
        if len(data) == 0:
            return False, "Data is empty"
        
        return True, None
    
    def generate_plot(self, query: str, data: pd.DataFrame, 
                     user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate interactive plot using LLM + Plotly"""
        
        try:
            # Step 1: Use LLM to interpret query and generate plot specification
            plot_spec = self._interpret_query_with_llm(query, data, context)
            
            # Step 2: Generate Plotly code based on specification
            plot_html = self._generate_plotly_code(plot_spec, data)
            plot_summary = self._build_plot_summary(plot_spec, data)
            
            return {
                'plot_html': plot_html,
                'plot_type': plot_spec.get('plot_type', 'interactive'),
                'metadata': {
                    'data_mode': self.data_mode,
                    'query': query,
                    'plot_spec': plot_spec,
                    'plot_summary': plot_summary
                }
            }
        except Exception as e:
            logger.error(f"Error generating plot: {e}")
            # Fallback to simple plot
            return self._generate_fallback_plot(data, query)
    
    def _interpret_query_with_llm(self, query: str, data: pd.DataFrame, 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Use OpenAI to interpret query and generate plot specification"""
        
        # Get data schema
        schema = {
            'columns': data.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'shape': data.shape,
            'sample': data.head(3).to_dict('records')
        }
        
        # Create prompt for LLM
        prompt = f"""You are a data visualization expert. Generate a Plotly plot specification based on the user's query.

User Query: "{query}"

Available Data Schema:
{json.dumps(schema, indent=2, default=str)}

Data Mode: sentiment_analysis
Available Columns: {', '.join(data.columns.tolist())}

Generate a JSON specification with:
- plot_type: one of {', '.join(self.supported_plot_types)}
- x_axis: column name for x-axis
- y_axis: column name for y-axis
- color_by: (optional) column for color encoding
- aggregation: (optional) aggregation method (mean, sum, count, etc.)
- chart_type: (bar, line, scatter, heatmap, box, histogram)
- title: descriptive title for the plot
- filters: (optional) any filters to apply

Respond ONLY with valid JSON, no markdown."""
        
        # Try OpenAI if available
        if OPENAI_AVAILABLE and openai:
            try:
                # Check if API key is configured
                if hasattr(openai, 'api_key') and openai.api_key:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a data visualization expert. Always respond with valid JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=500
                    )
                    
                    plot_spec = json.loads(response.choices[0].message.content)
                    return plot_spec
            except Exception as e:
                logger.warning(f"OpenAI API failed, using fallback: {e}")
        
        # Fallback to simple interpretation
        return self._fallback_interpretation(query, data)
    
    def _generate_plotly_code(self, plot_spec: Dict[str, Any], 
                             data: pd.DataFrame) -> str:
        """Generate Plotly HTML from specification"""
        
        try:
            plot_type = plot_spec.get('chart_type', 'bar')
            x_col = plot_spec.get('x_axis')
            y_col = plot_spec.get('y_axis')
            color_by = plot_spec.get('color_by')
            title = plot_spec.get('title', 'Interactive Plot')
            
            # Prepare data
            plot_data = data.copy()
            
            # Apply aggregation if specified
            if plot_spec.get('aggregation') and x_col and y_col:
                agg_func = plot_spec['aggregation']
                if agg_func in ['mean', 'sum', 'count', 'min', 'max']:
                    plot_data = plot_data.groupby(x_col)[y_col].agg(agg_func).reset_index()
            
            # Generate Plotly figure based on chart type
            fig = None
            
            if plot_type == 'bar':
                if color_by and color_by in plot_data.columns:
                    fig = px.bar(plot_data, x=x_col, y=y_col, color=color_by, title=title)
                elif x_col and y_col and x_col in plot_data.columns and y_col in plot_data.columns:
                    fig = px.bar(plot_data, x=x_col, y=y_col, title=title)
                else:
                    # Default: sentiment distribution
                    fig = self._create_sentiment_distribution(plot_data, title)
            
            elif plot_type == 'line':
                if color_by and color_by in plot_data.columns:
                    fig = px.line(plot_data, x=x_col, y=y_col, color=color_by, title=title)
                elif x_col and y_col and x_col in plot_data.columns and y_col in plot_data.columns:
                    fig = px.line(plot_data, x=x_col, y=y_col, title=title)
                else:
                    fig = self._create_sentiment_trend(plot_data, title)
            
            elif plot_type == 'scatter':
                if color_by and color_by in plot_data.columns:
                    fig = px.scatter(plot_data, x=x_col, y=y_col, color=color_by, title=title)
                elif x_col and y_col and x_col in plot_data.columns and y_col in plot_data.columns:
                    fig = px.scatter(plot_data, x=x_col, y=y_col, title=title)
                else:
                    fig = self._create_sentiment_distribution(plot_data, title)
            
            elif plot_type == 'heatmap':
                fig = self._create_sentiment_heatmap(plot_data, plot_spec, title)
            
            elif plot_type == 'box':
                if color_by and color_by in plot_data.columns:
                    fig = px.box(plot_data, x=x_col or color_by, y=y_col, color=color_by, title=title)
                elif x_col and y_col:
                    fig = px.box(plot_data, x=x_col, y=y_col, title=title)
                else:
                    fig = self._create_sentiment_distribution(plot_data, title)
            
            elif plot_type == 'histogram':
                col = x_col or y_col or 'sentiment'
                if col in plot_data.columns:
                    fig = px.histogram(plot_data, x=col, title=title)
                else:
                    fig = self._create_sentiment_distribution(plot_data, title)
            
            else:
                # Default to sentiment distribution
                fig = self._create_sentiment_distribution(plot_data, title)
            
            # Make it interactive
            fig.update_layout(
                hovermode='closest',
                template='plotly_white',
                height=600,
                title=title
            )
            
            # Convert to HTML
            return fig.to_html(include_plotlyjs='cdn', div_id="interactive-plot")
            
        except Exception as e:
            logger.error(f"Error generating Plotly code: {e}")
            return self._generate_simple_plot_html(data, title)

    def _build_plot_summary(self, plot_spec: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Build a rich, structured summary for chat grounding"""
        try:
            x_col = plot_spec.get('x_axis')
            y_col = plot_spec.get('y_axis')
            agg = plot_spec.get('aggregation')
            summary_data = {}

            if agg and x_col and y_col and x_col in data.columns and y_col in data.columns:
                grouped = data.groupby(x_col)[y_col].agg(agg).reset_index()
                grouped_sorted = grouped.sort_values(by=y_col, ascending=False).head(10)
                summary_data['top_values'] = grouped_sorted.to_dict('records')
                summary_text = f"Top 10 {y_col} by {x_col} using {agg} aggregation."
            else:
                sentiment_cols = [col for col in data.columns if 'sentiment' in col.lower()]
                if sentiment_cols:
                    sentiment_col = sentiment_cols[0]
                    summary_data['sentiment_stats'] = {
                        'min': float(data[sentiment_col].min()),
                        'max': float(data[sentiment_col].max()),
                        'mean': float(data[sentiment_col].mean()),
                        'median': float(data[sentiment_col].median())
                    }
                    summary_text = "Sentiment distribution summary statistics."
                else:
                    summary_text = "Plot summary generated from available data."

            return {
                'title': plot_spec.get('title', 'Sentiment Plot'),
                'plot_type': plot_spec.get('plot_type', 'interactive'),
                'description': plot_spec.get('chart_type', 'plot'),
                'data': summary_data,
                'summary_text': summary_text
            }
        except Exception as e:
            logger.warning(f"Failed to build plot summary: {e}")
            return {
                'title': plot_spec.get('title', 'Sentiment Plot'),
                'plot_type': plot_spec.get('plot_type', 'interactive'),
                'description': 'Plot summary unavailable',
                'data': {},
                'summary_text': 'Plot summary unavailable due to an internal error.'
            }
    
    def _create_sentiment_distribution(self, data: pd.DataFrame, title: str):
        """Create sentiment distribution histogram - clearly marked as TEXT data"""
        sentiment_col = [col for col in data.columns if 'sentiment' in col.lower()][0]
        fig = px.histogram(data, x=sentiment_col, title=f"📝 {title} (Text/Sentiment Data)", nbins=30)
        # Add annotation to make it clear this is text data
        fig.add_annotation(
            text="<b>TEXT DATA VISUALIZATION</b><br>Sentiment analysis of text content",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1
        )
        return fig
    
    def _create_sentiment_trend(self, data: pd.DataFrame, title: str):
        """Create sentiment trend over time - clearly marked as TEXT data"""
        sentiment_col = [col for col in data.columns if 'sentiment' in col.lower()][0]
        date_col = None
        
        # Look for date column
        for col in data.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col:
            # Convert to datetime if needed
            if data[date_col].dtype == 'object':
                data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            fig = px.line(data, x=date_col, y=sentiment_col, title=f"📝 {title} (Text/Sentiment Data)")
        else:
            # Use index as x-axis
            fig = px.line(data, y=sentiment_col, title=f"📝 {title} (Text/Sentiment Data)")
        
        # Add annotation
        fig.add_annotation(
            text="<b>TEXT DATA VISUALIZATION</b><br>Sentiment trend over time",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1
        )
        
        return fig
    
    def _create_sentiment_heatmap(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Create sentiment heatmap"""
        sentiment_col = [col for col in data.columns if 'sentiment' in col.lower()][0]
        
        # Try to create pivot table
        index_col = plot_spec.get('index_col') or plot_spec.get('x_axis')
        columns_col = plot_spec.get('columns_col') or plot_spec.get('color_by')
        
        if index_col and columns_col and index_col in data.columns and columns_col in data.columns:
            pivot_data = data.pivot_table(
                values=sentiment_col,
                index=index_col,
                columns=columns_col,
                aggfunc='mean'
            )
            fig = px.imshow(pivot_data, title=title, labels=dict(x=columns_col, y=index_col, color="Sentiment"))
        else:
            # Fallback to simple distribution
            fig = self._create_sentiment_distribution(data, title)
        
        return fig
    
    def _generate_simple_plot_html(self, data: pd.DataFrame, title: str) -> str:
        """Generate a simple fallback plot"""
        sentiment_col = [col for col in data.columns if 'sentiment' in col.lower()][0]
        fig = px.histogram(data, x=sentiment_col, title=title, nbins=30)
        fig.update_layout(height=600, template='plotly_white')
        return fig.to_html(include_plotlyjs='cdn', div_id="interactive-plot")
    
    def _fallback_interpretation(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Simple fallback when LLM fails"""
        query_lower = query.lower()
        
        # Find available columns
        sentiment_col = [col for col in data.columns if 'sentiment' in col.lower()][0] if any('sentiment' in col.lower() for col in data.columns) else None
        date_col = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()][0] if any('date' in col.lower() or 'time' in col.lower() for col in data.columns) else None
        asset_col = [col for col in data.columns if 'asset' in col.lower() or 'ticker' in col.lower() or 'symbol' in col.lower()][0] if any('asset' in col.lower() or 'ticker' in col.lower() or 'symbol' in col.lower() for col in data.columns) else None
        
        if 'time' in query_lower or 'date' in query_lower or 'trend' in query_lower or 'sentiment trend' in query_lower:
            return {
                'plot_type': 'sentiment_over_time',
                'chart_type': 'line',
                'x_axis': date_col or 'index',
                'y_axis': sentiment_col or data.columns[0],
                'title': 'Sentiment Over Time'
            }
        elif 'asset' in query_lower or 'ticker' in query_lower or 'symbol' in query_lower or 'compare' in query_lower or 'asset comparison' in query_lower:
            return {
                'plot_type': 'sentiment_by_asset',
                'chart_type': 'bar',
                'x_axis': asset_col or data.columns[0],
                'y_axis': sentiment_col or data.columns[1] if len(data.columns) > 1 else data.columns[0],
                'aggregation': 'mean',
                'title': 'Average Sentiment by Asset'
            }
        elif 'distribution' in query_lower or 'histogram' in query_lower:
            return {
                'plot_type': 'sentiment_distribution',
                'chart_type': 'histogram',
                'x_axis': sentiment_col or data.columns[0],
                'y_axis': None,
                'title': 'Sentiment Distribution'
            }
        else:
            # Default plot
            return {
                'plot_type': 'sentiment_distribution',
                'chart_type': 'histogram',
                'x_axis': sentiment_col or data.columns[0],
                'y_axis': None,
                'title': 'Sentiment Analysis'
            }
    
    def _generate_fallback_plot(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Generate a simple fallback plot when everything fails"""
        sentiment_col = [col for col in data.columns if 'sentiment' in col.lower()][0] if any('sentiment' in col.lower() for col in data.columns) else data.columns[0]
        
        fig = px.histogram(data, x=sentiment_col, title='Sentiment Distribution', nbins=30)
        fig.update_layout(height=600, template='plotly_white')
        plot_html = fig.to_html(include_plotlyjs='cdn', div_id="interactive-plot")
        
        return {
            'plot_html': plot_html,
            'plot_type': 'fallback',
            'metadata': {
                'data_mode': self.data_mode,
                'query': query,
                'note': 'Fallback plot generated'
            }
        }
    
    def get_supported_queries(self) -> list:
        """Return examples of supported query types"""
        return [
            "Show sentiment over time",
            "Compare sentiment by asset",
            "Show sentiment distribution",
            "Plot sentiment trends",
            "Create heatmap of sentiment by asset and date",
            "Show me a bar chart of average sentiment by ticker",
            "Generate a line plot of sentiment over time"
        ]

