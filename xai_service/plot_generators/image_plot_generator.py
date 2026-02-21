"""
Image Plot Generator - Generates interactive plots for image data analysis
"""

from .base_plot_generator import BasePlotGenerator
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, Tuple
import json
import logging
import base64
import io
import numpy as np

# Optional PIL import for image processing (if needed in future)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

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


class ImagePlotGenerator(BasePlotGenerator):
    """Plot generator for image data analysis"""
    
    def __init__(self):
        super().__init__('image')
        self.supported_plot_types = [
            'image_grid',
            'image_classification_results',
            'confusion_matrix',
            'image_statistics',  # Basic statistical profiling
            'image_statistics_enhanced',  # Enhanced image quality metrics
            'classification_distribution',  # Enhanced label analysis
            'class_distribution',  # Alias for classification_distribution
            'train_test_distribution',  # Train vs test split analysis
            'dataset_overview',  # Dataset summary card
            'class_balance_analysis',  # Class imbalance visualization
            'image_grid_by_class',  # Sample images organized by class
            'class_separability',  # Embedding-based class analysis
            'embedding_visualization',  # t-SNE/UMAP visualization
            'duplicate_detection',  # Perceptual hashing + embedding-based
            'anomaly_detection',  # Pre-trained model based
            'class_similarity_matrix',  # NEW: Inter-class similarity using ResNet18
            'nearest_neighbors',  # NEW: Nearest neighbor analysis using ResNet18
            'image_metadata_analysis',
            'bar',
            'scatter',
            'heatmap',
            'histogram'
        ]
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Check if data has image-related columns"""
        # Check for image columns (base64, file paths, or image metadata)
        image_cols = self._find_image_columns(data)
        
        if not image_cols:
            # Check if we have image metadata (labels, predictions, etc.)
            metadata_cols = [col for col in data.columns 
                           if any(term in col.lower() for term in ['label', 'class', 'prediction', 'category', 'score'])]
            if not metadata_cols:
                return False, "Data missing image columns or image metadata"
        
        # Check if we have at least some data
        if len(data) == 0:
            return False, "Data is empty"
        
        return True, None
    
    def _find_image_columns(self, data: pd.DataFrame) -> list:
        """Find columns that contain image data"""
        image_cols = []
        
        for col in data.columns:
            col_lower = col.lower()
            # Check column name
            if any(term in col_lower for term in ['image', 'img', 'picture', 'photo', 'file_path', 'path']):
                image_cols.append(col)
                continue
            
            # Check if column contains base64 data
            if data[col].dtype == 'object':
                sample = data[col].dropna().head(5)
                if len(sample) > 0:
                    # Check if values look like base64
                    first_val = str(sample.iloc[0])
                    if (first_val.startswith('data:image') or 
                        (len(first_val) > 100 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in first_val[:100]))):
                        image_cols.append(col)
                        continue
                    
                    # Check if values look like file paths
                    if any(ext in first_val.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']):
                        image_cols.append(col)
        
        return image_cols
    
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
                    'columns_used': plot_spec.get('columns_used', []),
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
        image_cols = self._find_image_columns(data)
        metadata_cols = [col for col in data.columns 
                        if any(term in col.lower() for term in ['label', 'class', 'prediction', 'category', 'score', 'confidence'])]
        
        schema = {
            'columns': data.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'shape': data.shape,
            'image_columns': image_cols,
            'metadata_columns': metadata_cols,
            'sample': data.head(3).to_dict('records') if len(data) > 0 else []
        }
        
        # Create prompt for LLM
        prompt = f"""You are a data visualization expert specializing in IMAGE data analysis (NOT text data). 
This dataset contains actual IMAGES, not text content.

User Query: "{query}"

Available Data Schema:
{json.dumps(schema, indent=2, default=str)}

Data Mode: IMAGE_ANALYSIS (this is image data, not text/sentiment data)
Image Columns: {', '.join(image_cols) if image_cols else 'None found'}
Metadata Columns: {', '.join(metadata_cols) if metadata_cols else 'None found'}

IMPORTANT - DATA EXPLORATION PLOTS (prioritize these for data analysis):
- If user asks about "class distribution", "label distribution", "show distribution" → use chart_type: "class_distribution"
- If user asks about "train test", "split", "train vs test" → use chart_type: "train_test_distribution"
- If user asks about "dataset overview", "dataset summary", "overview" → use chart_type: "dataset_overview"
- If user asks about "class balance", "balance", "imbalance" → use chart_type: "class_balance_analysis"
- If user asks about "image statistics", "statistics", "dimensions", "aspect ratio", "file size" → use chart_type: "image_statistics_enhanced"
- If user asks to "show images by class", "sample images", "display images by class" → use chart_type: "image_grid_by_class"

MODEL PERFORMANCE PLOTS (XAI - separate from data exploration):
- If user asks to "show images", "display images", "see images", "image grid" → use chart_type: "image_grid"
- If user asks about "embeddings", "t-sne", "umap", "visualization", "clusters" → use chart_type: "embedding_visualization"
- If user asks about "separability", "class distance" (in context of model) → use chart_type: "class_separability"
- If user asks about "duplicates", "similar images", "redundancy" → use chart_type: "duplicate_detection"
- If user asks about "anomaly", "outlier", "unusual", "abnormal" → use chart_type: "anomaly_detection"
- If user asks about "confusion matrix", "accuracy", "misclassification" → use chart_type: "confusion_matrix"
- Always remember this is IMAGE data, not text data
- Distinguish between DATA EXPLORATION (analyzing the dataset) and MODEL PERFORMANCE (analyzing predictions)

Generate a JSON specification with:
- plot_type: one of {', '.join(self.supported_plot_types)}
- chart_type: (image_grid for showing actual images, bar/scatter/heatmap/histogram for metadata, confusion_matrix for classification)
- x_axis: column name for x-axis (if applicable, usually for metadata plots)
- y_axis: column name for y-axis (if applicable, usually for metadata plots)
- color_by: (optional) column for color encoding
- aggregation: (optional) aggregation method (mean, sum, count, etc.)
- title: descriptive title for the plot (include "Image Data" or "📸" to indicate images)
- columns_used: list of columns that will be used

Respond ONLY with valid JSON, no markdown."""
        
        # Try OpenAI if available
        if OPENAI_AVAILABLE and openai:
            try:
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
            title = plot_spec.get('title', 'Image Data Analysis')
            
            # Prepare data
            plot_data = data.copy()
            
            # Apply aggregation if specified
            if plot_spec.get('aggregation') and x_col and y_col:
                agg_func = plot_spec.get('aggregation')
                if agg_func in ['mean', 'sum', 'count', 'min', 'max']:
                    plot_data = plot_data.groupby(x_col)[y_col].agg(agg_func).reset_index()
            
            # Generate Plotly figure based on chart type
            fig = None
            
            if plot_type == 'confusion_matrix':
                fig = self._create_confusion_matrix(plot_data, plot_spec, title)
            
            elif plot_type == 'image_grid':
                grid_result = self._create_image_grid(plot_data, plot_spec, title)
                if isinstance(grid_result, dict) and grid_result.get('type') == 'html_grid':
                    # Return custom HTML for image grid
                    return self._wrap_html_in_plotly(grid_result['html'], grid_result['title'])
                elif isinstance(grid_result, go.Figure):
                    fig = grid_result
                else:
                    # Fallback
                    fig = self._create_classification_distribution(plot_data, plot_spec, title)
            
            elif plot_type == 'classification_distribution' or plot_type == 'class_distribution':
                fig = self._create_classification_distribution(plot_data, plot_spec, title)
            
            elif plot_type == 'train_test_distribution':
                fig = self._create_train_test_distribution(plot_data, plot_spec, title)
            
            elif plot_type == 'dataset_overview':
                fig = self._create_dataset_overview(plot_data, plot_spec, title)
            
            elif plot_type == 'class_balance_analysis':
                fig = self._create_class_balance_analysis(plot_data, plot_spec, title)
            
            elif plot_type == 'image_statistics_enhanced':
                # Use enhanced version if available, fallback to regular
                fig = self._create_image_statistics(plot_data, plot_spec, title)
            
            elif plot_type == 'image_grid_by_class':
                grid_result = self._create_image_grid_by_class(plot_data, plot_spec, title)
                if isinstance(grid_result, dict) and grid_result.get('type') == 'html_grid':
                    return self._wrap_html_in_plotly(grid_result['html'], grid_result['title'])
                elif isinstance(grid_result, go.Figure):
                    fig = grid_result
                else:
                    fig = self._create_classification_distribution(plot_data, plot_spec, title)
            
            elif plot_type == 'bar':
                if color_by and color_by in plot_data.columns:
                    fig = px.bar(plot_data, x=x_col, y=y_col, color=color_by, title=title)
                elif x_col and y_col and x_col in plot_data.columns and y_col in plot_data.columns:
                    fig = px.bar(plot_data, x=x_col, y=y_col, title=title)
                else:
                    fig = self._create_classification_distribution(plot_data, plot_spec, title)
            
            elif plot_type == 'scatter':
                if color_by and color_by in plot_data.columns:
                    fig = px.scatter(plot_data, x=x_col, y=y_col, color=color_by, title=title)
                elif x_col and y_col and x_col in plot_data.columns and y_col in plot_data.columns:
                    fig = px.scatter(plot_data, x=x_col, y=y_col, title=title)
                else:
                    fig = self._create_classification_distribution(plot_data, plot_spec, title)
            
            elif plot_type == 'heatmap':
                fig = self._create_heatmap(plot_data, plot_spec, title)
            
            elif plot_type == 'histogram':
                col = x_col or y_col
                if col and col in plot_data.columns:
                    fig = px.histogram(plot_data, x=col, title=title)
                else:
                    fig = self._create_classification_distribution(plot_data, plot_spec, title)
            
            elif plot_type == 'image_statistics':
                fig = self._create_image_statistics(plot_data, plot_spec, title)
            
            elif plot_type == 'embedding_visualization':
                fig = self._create_embedding_visualization(plot_data, plot_spec, title)
            
            elif plot_type == 'class_separability':
                fig = self._create_class_separability(plot_data, plot_spec, title)
            
            elif plot_type == 'duplicate_detection':
                fig = self._create_duplicate_detection(plot_data, plot_spec, title)
            
            elif plot_type == 'anomaly_detection':
                fig = self._create_anomaly_detection(plot_data, plot_spec, title)
            
            elif plot_type == 'class_similarity_matrix':
                fig = self._create_class_similarity_matrix(plot_data, plot_spec, title)
            
            elif plot_type == 'nearest_neighbors':
                fig = self._create_nearest_neighbors(plot_data, plot_spec, title)
            
            else:
                # Default to classification distribution
                fig = self._create_classification_distribution(plot_data, plot_spec, title)
            
            # Make it interactive and clearly mark as IMAGE data
            if fig is not None:
                # Add image data indicator to all plots
                fig.add_annotation(
                    text="<b>📸 IMAGE DATA</b>",
                    xref="paper", yref="paper",
                    x=0.98, y=0.02,
                    showarrow=False,
                    align="right",
                    bgcolor="rgba(52, 152, 219, 0.2)",
                    bordercolor="rgba(52, 152, 219, 0.5)",
                    borderwidth=1,
                    font=dict(size=12, color="#2980b9")
                )
                fig.update_layout(
                    hovermode='closest',
                    template='plotly_white',
                    height=600,
                    title=f"📸 {title}"
                )
            
            # Convert to HTML
            return fig.to_html(include_plotlyjs='cdn', div_id="interactive-plot")
            
        except Exception as e:
            logger.error(f"Error generating Plotly code: {e}")
            return self._generate_simple_plot_html(data, title)

    def _build_plot_summary(self, plot_spec: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Build a rich, structured summary for chat grounding - enhanced for data exploration"""
        try:
            image_cols = self._find_image_columns(data)
            metadata_cols = [col for col in data.columns 
                            if any(term in col.lower() for term in ['label', 'class', 'prediction', 'category', 'score', 'confidence'])]
            
            plot_type = plot_spec.get('chart_type', plot_spec.get('plot_type', 'interactive'))
            summary_data = {
                'image_columns': image_cols[:5],
                'metadata_columns': metadata_cols[:5],
                'row_count': int(len(data)),
                'plot_type': plot_type
            }

            label_col = None
            split_col = None
            for col in data.columns:
                col_lower = col.lower()
                if col_lower in ['label', 'digit']:
                    label_col = col
                elif col_lower == 'split':
                    split_col = col

            # Enhanced metadata based on plot type
            if plot_type in ['class_distribution', 'classification_distribution']:
                if label_col:
                    class_counts = data[label_col].value_counts().sort_index()
                    total = len(data)
                    percentages = (class_counts / total * 100).round(1)
                    summary_data['class_counts'] = class_counts.to_dict()
                    summary_data['class_percentages'] = {str(k): float(v) for k, v in percentages.to_dict().items()}
                    summary_data['total_classes'] = int(len(class_counts))
                    summary_text = f"Class distribution: {len(class_counts)} classes, {total} total images."
                else:
                    summary_text = "Class distribution plot - label column not found."
            
            elif plot_type == 'train_test_distribution':
                if split_col:
                    split_counts = data[split_col].value_counts()
                    total = len(data)
                    split_percentages = (split_counts / total * 100).round(1)
                    summary_data['split_counts'] = split_counts.to_dict()
                    summary_data['split_percentages'] = {str(k): float(v) for k, v in split_percentages.to_dict().items()}
                    train_count = split_counts.get('train', 0)
                    test_count = split_counts.get('test', 0)
                    summary_text = f"Train/Test split: {train_count} train ({split_percentages.get('train', 0):.1f}%), {test_count} test ({split_percentages.get('test', 0):.1f}%)."
                else:
                    summary_text = "Train/Test distribution plot - split column not found."
            
            elif plot_type == 'dataset_overview':
                summary_data['total_images'] = int(len(data))
                summary_data['num_columns'] = int(len(data.columns))
                summary_data['column_names'] = data.columns.tolist()[:10]
                if label_col:
                    summary_data['num_classes'] = int(len(data[label_col].unique()))
                    summary_data['classes'] = sorted(data[label_col].unique().tolist())
                if split_col:
                    summary_data['has_split'] = True
                    summary_data['split_counts'] = data[split_col].value_counts().to_dict()
                summary_text = f"Dataset overview: {len(data)} images, {len(data.columns)} columns."
            
            elif plot_type == 'class_balance_analysis':
                if label_col:
                    class_counts = data[label_col].value_counts()
                    max_count = int(class_counts.max())
                    min_count = int(class_counts.min())
                    imbalance_ratio = float(max_count / min_count) if min_count > 0 else float('inf')
                    summary_data['class_counts'] = class_counts.to_dict()
                    summary_data['imbalance_ratio'] = imbalance_ratio
                    summary_data['max_class_count'] = max_count
                    summary_data['min_class_count'] = min_count
                    summary_text = f"Class balance: ratio {imbalance_ratio:.2f} (max: {max_count}, min: {min_count})."
                else:
                    summary_text = "Class balance analysis - label column not found."
            
            elif plot_type in ['image_statistics', 'image_statistics_enhanced']:
                summary_data['has_image_stats'] = True
                summary_text = "Image statistics including dimensions, aspect ratios, and quality metrics."
            
            else:
                # Default summary
                if label_col:
                    class_counts = data[label_col].value_counts().head(10)
                    summary_data['top_classes'] = class_counts.to_dict()
                    summary_data['class_distribution'] = class_counts.to_dict()
                    summary_text = f"Top 10 classes by image count. Total: {len(data)} images."
                else:
                    summary_text = f"Image metadata summary. Total: {len(data)} images, {len(data.columns)} columns."

            return {
                'title': plot_spec.get('title', 'Image Plot'),
                'plot_type': plot_spec.get('plot_type', 'interactive'),
                'description': plot_spec.get('chart_type', 'plot'),
                'data': summary_data,
                'summary_text': summary_text
            }
        except Exception as e:
            logger.warning(f"Failed to build plot summary: {e}")
            return {
                'title': plot_spec.get('title', 'Image Plot'),
                'plot_type': plot_spec.get('plot_type', 'interactive'),
                'description': 'Plot summary unavailable',
                'data': {},
                'summary_text': 'Plot summary unavailable due to an internal error.'
            }
    
    def _create_confusion_matrix(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Create confusion matrix for image classification - clearly marked as IMAGE data"""
        # Find label and prediction columns
        label_col = None
        pred_col = None
        
        for col in data.columns:
            col_lower = col.lower()
            if 'label' in col_lower or 'true' in col_lower or 'actual' in col_lower:
                label_col = col
            if 'prediction' in col_lower or 'pred' in col_lower or 'predicted' in col_lower:
                pred_col = col
        
        if label_col and pred_col:
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(data[label_col], data[pred_col])
            
            # Get unique labels
            labels = sorted(list(set(data[label_col].unique()) | set(data[pred_col].unique())))
            
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=labels,
                y=labels,
                title=f"📸 {title} - Image Classification Confusion Matrix",
                color_continuous_scale='Blues',
                text_auto=True
            )
            # Add annotation
            fig.add_annotation(
                text="<b>📸 IMAGE DATA</b><br>Confusion matrix for image classification",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                align="left",
                bgcolor="rgba(52, 152, 219, 0.2)",
                bordercolor="rgba(52, 152, 219, 0.5)",
                borderwidth=1
            )
            return fig
        else:
            # Fallback to classification distribution
            return self._create_classification_distribution(data, plot_spec, title)
    
    def _create_image_grid(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Create image grid visualization with actual images displayed"""
        image_cols = self._find_image_columns(data)
        metadata_cols = [col for col in data.columns 
                        if any(term in col.lower() for term in ['label', 'class', 'category', 'digit'])]
        
        if not image_cols:
            # Fallback to metadata visualization
            return self._create_classification_distribution(data, plot_spec, title)
        
        image_col = image_cols[0]
        label_col = metadata_cols[0] if metadata_cols else None
        
        # Limit to reasonable number of images for display (max 20)
        max_images = min(20, len(data))
        
        # Group by label if available for better organization
        if label_col:
            # Show sample images from each class
            grouped = data.groupby(label_col)
            html_images = []
            
            for label, group in grouped:
                for idx, (_, row) in enumerate(group.head(3).iterrows()):  # Max 3 per class
                    image_data = str(row[image_col])
                    html_images.append({
                        'image': image_data,
                        'label': row[label_col],
                        'id': f"img_{len(html_images)}"
                    })
                    if len(html_images) >= max_images:
                        break
                if len(html_images) >= max_images:
                    break
        else:
            # Just take first N images
            html_images = []
            for idx, (_, row) in enumerate(data.head(max_images).iterrows()):
                image_data = str(row[image_col])
                html_images.append({
                    'image': image_data,
                    'label': None,
                    'id': f"img_{idx}"
                })
        
        # Create HTML-based image grid (will be embedded in Plotly HTML)
        n_cols = 5
        grid_html = f"""
        <div style="padding: 20px; background: white; font-family: Arial, sans-serif;">
            <h3 style="text-align: center; margin-bottom: 20px; color: #2c3e50;">{title}</h3>
            <p style="text-align: center; color: #7f8c8d; margin-bottom: 20px;">📸 Image Grid - Showing {len(html_images)} sample images from dataset</p>
            <div style="display: grid; grid-template-columns: repeat({n_cols}, 1fr); gap: 15px; max-width: 1200px; margin: 0 auto;">
        """
        
        for img_info in html_images:
            label_text = f"<div style='text-align: center; font-weight: bold; margin-top: 8px; color: #34495e; font-size: 14px;'>Label: {img_info['label']}</div>" if img_info['label'] is not None else ""
            grid_html += f"""
                <div style="border: 2px solid #bdc3c7; border-radius: 8px; padding: 10px; background: #ecf0f1; text-align: center; transition: transform 0.2s;">
                    <img src="{img_info['image']}" 
                         style="max-width: 100%; height: auto; max-height: 120px; border-radius: 4px; display: block; margin: 0 auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" 
                         alt="Image {img_info['id']}"
                         onerror="this.style.display='none'; this.parentElement.innerHTML='<div style=\\'padding: 20px; color: #e74c3c;\\'>⚠️ Image failed to load</div>';">
                    {label_text}
                </div>
            """
        
        grid_html += """
            </div>
            <div style="text-align: center; margin-top: 20px; color: #95a5a6; font-size: 12px;">
                <p>💡 This is an <strong>IMAGE DATA</strong> visualization showing actual images from your dataset</p>
            </div>
        </div>
        """
        
        # Return custom HTML that will be embedded
        # We'll modify the _generate_plotly_code to handle this special case
        return {'type': 'html_grid', 'html': grid_html, 'title': title}
    
    def _create_classification_distribution(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Create distribution of image classifications - clearly marked as IMAGE data"""
        # Find classification/label column (prefer label over prediction for data plots)
        class_col = None
        for col in data.columns:
            col_lower = col.lower()
            # For data plots, prefer 'label' or 'digit' over 'prediction'
            if col_lower in ['label', 'digit']:
                class_col = col
                break
            elif any(term in col_lower for term in ['class', 'category']):
                class_col = col
                break
        
        # Fallback to prediction if label not found (but this is less ideal for data plots)
        if not class_col:
            for col in data.columns:
                col_lower = col.lower()
                if 'prediction' in col_lower:
                    class_col = col
                    break
        
        if class_col:
            class_counts = data[class_col].value_counts().head(30).sort_index()
            total = len(data)
            percentages = (class_counts / total * 100).round(1)
            
            # Create subplot with counts and percentages
            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Class Distribution (Counts)', 'Class Distribution (Percentages)'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Counts bar chart
            fig.add_trace(
                go.Bar(
                    x=class_counts.index.astype(str),
                    y=class_counts.values,
                    name='Count',
                    marker_color='steelblue',
                    text=class_counts.values,
                    textposition='outside',
                    hovertemplate='Class: %{x}<br>Count: %{y}<br><extra></extra>'
                ),
                row=1, col=1
            )
            
            # Percentages bar chart
            fig.add_trace(
                go.Bar(
                    x=percentages.index.astype(str),
                    y=percentages.values,
                    name='Percentage',
                    marker_color='lightblue',
                    text=[f'{p:.1f}%' for p in percentages.values],
                    textposition='outside',
                    hovertemplate='Class: %{x}<br>Percentage: %{text}<br><extra></extra>'
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Class/Label", row=1, col=1)
            fig.update_yaxes(title_text="Number of Images", row=1, col=1)
            fig.update_xaxes(title_text="Class/Label", row=1, col=2)
            fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
            
            fig.update_layout(
                title=f"📸 {title} - Class Distribution (Data Analysis)",
                height=500,
                template='plotly_white',
                showlegend=False
            )
            
            # Add annotation to make it clear this is data analysis
            fig.add_annotation(
                text=f"<b>📸 DATA EXPLORATION</b><br>Total Images: {total}<br>Classes: {len(class_counts)}<br><i>Analyzing dataset distribution, not model performance</i>",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                align="left",
                bgcolor="rgba(52, 152, 219, 0.15)",
                bordercolor="rgba(52, 152, 219, 0.5)",
                borderwidth=2,
                font=dict(size=11)
            )
        else:
            # Fallback: show data shape
            fig = go.Figure()
            fig.add_annotation(
                text=f"<b>IMAGE DATASET OVERVIEW</b><br>Total Images: {len(data)}<br>Columns: {len(data.columns)}<br><br>📸 This is image data, not text data",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16),
                bgcolor="rgba(52, 152, 219, 0.1)",
                bordercolor="rgba(52, 152, 219, 0.5)",
                borderwidth=2
            )
            fig.update_layout(title=f"📸 {title} - Image Data", height=400)
        
        return fig
    
    def _create_class_distribution(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Alias for _create_classification_distribution - enhanced class distribution"""
        return self._create_classification_distribution(data, plot_spec, title)
    
    def _add_image_data_indicator(self, fig):
        """Add IMAGE DATA indicator annotation to plotly figure"""
        if fig is not None:
            fig.add_annotation(
                text="<b>📸 IMAGE DATA</b>",
                xref="paper", yref="paper",
                x=0.98, y=0.02,
                showarrow=False,
                align="right",
                bgcolor="rgba(52, 152, 219, 0.2)",
                bordercolor="rgba(52, 152, 219, 0.5)",
                borderwidth=1,
                font=dict(size=12, color="#2980b9")
            )
        return fig
    
    def _create_heatmap(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Create heatmap for image metadata"""
        # Try to create correlation heatmap for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                title=title,
                labels=dict(x="Feature", y="Feature", color="Correlation"),
                color_continuous_scale='RdBu',
                aspect="auto"
            )
        else:
            # Fallback to classification distribution
            return self._create_classification_distribution(data, plot_spec, title)
        
        return fig
    
    def _generate_simple_plot_html(self, data: pd.DataFrame, title: str) -> str:
        """Generate a simple fallback plot"""
        class_col = None
        for col in data.columns:
            if any(term in col.lower() for term in ['label', 'class', 'category']):
                class_col = col
                break
        
        if class_col:
            class_counts = data[class_col].value_counts().head(20)
            fig = px.bar(x=class_counts.index, y=class_counts.values, title=title)
        else:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Total Images: {len(data)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            fig.update_layout(title=title)
        
        fig.update_layout(height=600, template='plotly_white')
        return fig.to_html(include_plotlyjs='cdn', div_id="interactive-plot")
    
    def _fallback_interpretation(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Simple fallback when LLM fails - prioritize data exploration queries"""
        query_lower = query.lower()
        
        # Find available columns
        image_cols = self._find_image_columns(data)
        class_col = None
        for col in data.columns:
            if any(term in col.lower() for term in ['label', 'class', 'category', 'digit']):
                class_col = col
                break
        
        # DATA EXPLORATION QUERIES (prioritize these)
        if 'class distribution' in query_lower or 'label distribution' in query_lower or ('distribution' in query_lower and 'class' in query_lower):
            return {
                'plot_type': 'class_distribution',
                'chart_type': 'class_distribution',
                'title': 'Class Distribution - Data Analysis'
            }
        elif 'train' in query_lower and 'test' in query_lower and ('split' in query_lower or 'distribution' in query_lower):
            return {
                'plot_type': 'train_test_distribution',
                'chart_type': 'train_test_distribution',
                'title': 'Train/Test Split Distribution - Data Analysis'
            }
        elif 'dataset overview' in query_lower or 'dataset summary' in query_lower or ('overview' in query_lower and 'dataset' in query_lower):
            return {
                'plot_type': 'dataset_overview',
                'chart_type': 'dataset_overview',
                'title': 'Dataset Overview - Data Analysis'
            }
        elif 'class balance' in query_lower or 'balance' in query_lower or 'imbalance' in query_lower:
            return {
                'plot_type': 'class_balance_analysis',
                'chart_type': 'class_balance_analysis',
                'title': 'Class Balance Analysis - Data Exploration'
            }
        elif 'image statistics' in query_lower or ('statistics' in query_lower and 'image' in query_lower):
            return {
                'plot_type': 'image_statistics_enhanced',
                'chart_type': 'image_statistics_enhanced',
                'title': 'Image Statistics - Data Analysis'
            }
        elif ('sample' in query_lower or 'show' in query_lower or 'display' in query_lower) and 'image' in query_lower and 'class' in query_lower:
            return {
                'plot_type': 'image_grid_by_class',
                'chart_type': 'image_grid_by_class',
                'title': 'Sample Images by Class - Data Exploration'
            }
        
        # MODEL PERFORMANCE QUERIES (XAI - keep separate)
        elif 'confusion' in query_lower or 'matrix' in query_lower:
            return {
                'plot_type': 'confusion_matrix',
                'chart_type': 'confusion_matrix',
                'title': 'Confusion Matrix'
            }
        elif 'statistic' in query_lower or 'dimension' in query_lower or 'aspect ratio' in query_lower or 'file size' in query_lower:
            return {
                'plot_type': 'image_statistics',
                'chart_type': 'image_statistics',
                'title': 'Image Statistics - Dimensions, Aspect Ratios, File Sizes'
            }
        elif 'embedding' in query_lower or 't-sne' in query_lower or 'umap' in query_lower or ('cluster' in query_lower and 'visual' in query_lower):
            return {
                'plot_type': 'embedding_visualization',
                'chart_type': 'embedding_visualization',
                'title': 'Image Embedding Visualization (t-SNE)'
            }
        elif 'separability' in query_lower or ('class' in query_lower and 'distance' in query_lower) or 'imbalance' in query_lower:
            return {
                'plot_type': 'class_separability',
                'chart_type': 'class_separability',
                'title': 'Class Separability Analysis'
            }
        elif 'duplicate' in query_lower or ('similar' in query_lower and 'image' in query_lower) or 'redundancy' in query_lower:
            return {
                'plot_type': 'duplicate_detection',
                'chart_type': 'duplicate_detection',
                'title': 'Duplicate Image Detection'
            }
        elif 'anomaly' in query_lower or 'outlier' in query_lower or ('unusual' in query_lower and 'image' in query_lower) or 'abnormal' in query_lower:
            return {
                'plot_type': 'anomaly_detection',
                'chart_type': 'anomaly_detection',
                'title': 'Anomaly Detection (Pre-trained Model)'
            }
        elif 'grid' in query_lower or ('show' in query_lower and 'image' in query_lower) or 'display image' in query_lower:
            return {
                'plot_type': 'image_grid',
                'chart_type': 'image_grid',
                'title': 'Image Grid - Sample Images from Dataset'
            }
        elif 'sample' in query_lower and 'image' in query_lower:
            return {
                'plot_type': 'image_grid',
                'chart_type': 'image_grid',
                'title': 'Sample Images by Category'
            }
        elif 'distribution' in query_lower or 'histogram' in query_lower or 'count' in query_lower:
            return {
                'plot_type': 'classification_distribution',
                'chart_type': 'bar',
                'x_axis': class_col or data.columns[0] if len(data.columns) > 0 else None,
                'y_axis': None,
                'title': 'Image Classification Distribution (Metadata Analysis)'
            }
        elif 'heatmap' in query_lower or 'correlation' in query_lower:
            return {
                'plot_type': 'image_metadata_analysis',
                'chart_type': 'heatmap',
                'title': 'Image Metadata Correlation'
            }
        else:
            # Default: show image grid if images available, otherwise metadata
            image_cols = self._find_image_columns(data)
            if image_cols:
                return {
                    'plot_type': 'image_grid',
                    'chart_type': 'image_grid',
                    'title': 'Image Grid - Sample Images'
                }
            else:
                return {
                    'plot_type': 'classification_distribution',
                    'chart_type': 'bar',
                    'x_axis': class_col or data.columns[0] if len(data.columns) > 0 else None,
                    'y_axis': None,
                    'title': 'Image Data Analysis (Metadata)'
                }
    
    def _generate_fallback_plot(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Generate a simple fallback plot when everything fails - ALWAYS generates a valid plot"""
        class_col = None
        for col in data.columns:
            if any(term in col.lower() for term in ['label', 'class', 'category']):
                class_col = col
                break
        
        if class_col:
            class_counts = data[class_col].value_counts().head(20)
            fig = px.bar(x=class_counts.index, y=class_counts.values, title='Image Classification Distribution')
        else:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Total Images: {len(data)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            fig.update_layout(title='Image Data Overview')
        
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
    
    def _wrap_html_in_plotly(self, html_content: str, title: str) -> str:
        """Wrap custom HTML content in Plotly-compatible HTML structure"""
        # Create a complete HTML page with the image grid that can be embedded
        # Use iframe or direct embedding approach
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: #f8f9fa;
        }}
        .image-grid-container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
    </style>
</head>
<body>
    <div class="image-grid-container">
        {html_content}
    </div>
</body>
</html>
        """
        return full_html
    
    def _create_image_statistics(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Create basic statistical profiling: dimensions, aspect ratios, file sizes"""
        image_cols = self._find_image_columns(data)
        
        if not image_cols or not PIL_AVAILABLE:
            # Fallback to metadata statistics
            return self._create_classification_distribution(data, plot_spec, title)
        
        # Try to extract image statistics from base64 images
        stats = {
            'widths': [],
            'heights': [],
            'aspect_ratios': [],
            'file_sizes': [],
            'color_channels': {'r_mean': [], 'g_mean': [], 'b_mean': [], 'r_std': [], 'g_std': [], 'b_std': []},
            'saturation': [],
            'quality': {'sharpness': [], 'contrast': []}
        }
        
        sample_size = min(100, len(data))  # Limit to first 100 for performance
        image_col = image_cols[0]
        
        # Try to import scipy for sharpness calculation
        try:
            from scipy import ndimage
            SCIPY_AVAILABLE = True
        except ImportError:
            SCIPY_AVAILABLE = False
            logger.debug("scipy not available, skipping sharpness calculation")
        
        for idx, row in data.head(sample_size).iterrows():
            try:
                image_data = str(row[image_col])
                if image_data.startswith('data:image'):
                    # Extract base64 part
                    base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                    img_bytes = base64.b64decode(base64_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    width, height = img.size
                    aspect_ratio = width / height if height > 0 else 0
                    file_size = len(img_bytes)
                    
                    stats['widths'].append(width)
                    stats['heights'].append(height)
                    stats['aspect_ratios'].append(aspect_ratio)
                    stats['file_sizes'].append(file_size)
                    
                    # Color channel statistics (convert to RGB if needed)
                    img_rgb = img.convert('RGB')
                    img_array = np.array(img_rgb)
                    
                    # Calculate mean and std for each channel
                    r_mean = np.mean(img_array[:, :, 0])
                    g_mean = np.mean(img_array[:, :, 1])
                    b_mean = np.mean(img_array[:, :, 2])
                    r_std = np.std(img_array[:, :, 0])
                    g_std = np.std(img_array[:, :, 1])
                    b_std = np.std(img_array[:, :, 2])
                    
                    stats['color_channels']['r_mean'].append(r_mean)
                    stats['color_channels']['g_mean'].append(g_mean)
                    stats['color_channels']['b_mean'].append(b_mean)
                    stats['color_channels']['r_std'].append(r_std)
                    stats['color_channels']['g_std'].append(g_std)
                    stats['color_channels']['b_std'].append(b_std)
                    
                    # Calculate saturation (HSV color space)
                    img_hsv = img.convert('HSV')
                    hsv_array = np.array(img_hsv)
                    saturation = np.mean(hsv_array[:, :, 1])  # S channel
                    stats['saturation'].append(saturation)
                    
                    # Data quality metrics
                    # Convert to grayscale for quality metrics
                    gray = img.convert('L')
                    gray_array = np.array(gray)
                    
                    # Sharpness: Laplacian variance (higher = sharper)
                    if SCIPY_AVAILABLE:
                        try:
                            laplacian = ndimage.laplace(gray_array)
                            sharpness = np.var(laplacian)
                            stats['quality']['sharpness'].append(sharpness)
                        except Exception as e:
                            logger.debug(f"Could not calculate sharpness: {e}")
                    
                    # Contrast: Standard deviation of pixel intensities
                    contrast = np.std(gray_array)
                    stats['quality']['contrast'].append(contrast)
                    
            except Exception as e:
                logger.debug(f"Could not process image {idx}: {e}")
                continue
        
        if not stats['widths']:
            return self._create_classification_distribution(data, plot_spec, title)
        
        # Determine number of rows based on available data
        has_color_stats = len(stats['color_channels']['r_mean']) > 0
        has_quality_stats = len(stats['quality']['contrast']) > 0
        
        # Create comprehensive subplots
        if has_color_stats and has_quality_stats:
            # Full statistics with color and quality
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=('Image Dimensions', 'Aspect Ratio Distribution', 
                              'File Size Distribution', 'Color Channel Means (RGB)',
                              'Color Channel Std Dev', 'Saturation Distribution',
                              'Sharpness Distribution', 'Contrast Distribution', 'Statistics Summary'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}, {"type": "table"}]]
            )
        elif has_color_stats:
            # Statistics with color but no quality
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Image Dimensions', 'Aspect Ratio Distribution', 
                              'File Size Distribution', 'Color Channel Means (RGB)',
                              'Color Channel Std Dev', 'Statistics Summary'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}, {"type": "table"}]]
            )
        else:
            # Basic statistics only
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Image Dimensions Distribution', 'Aspect Ratio Distribution', 
                              'File Size Distribution', 'Dimension Statistics'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "table"}]]
            )
        
        # Width/Height scatter
        fig.add_trace(
            go.Scatter(x=stats['widths'], y=stats['heights'], mode='markers',
                      marker=dict(size=5, opacity=0.6), name='Images'),
            row=1, col=1
        )
        fig.update_xaxes(title_text="Width (px)", row=1, col=1)
        fig.update_yaxes(title_text="Height (px)", row=1, col=1)
        
        # Aspect ratio histogram
        fig.add_trace(
            go.Histogram(x=stats['aspect_ratios'], nbinsx=20, name='Aspect Ratios', marker_color='steelblue'),
            row=1, col=2
        )
        fig.update_xaxes(title_text="Aspect Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        # File size histogram
        file_sizes_mb = [s / (1024*1024) for s in stats['file_sizes']]
        fig.add_trace(
            go.Histogram(x=file_sizes_mb, nbinsx=20, name='File Sizes', marker_color='lightblue'),
            row=1, col=3 if (has_color_stats or has_quality_stats) else 1
        )
        col_idx = 3 if (has_color_stats or has_quality_stats) else 1
        fig.update_xaxes(title_text="File Size (MB)", row=1, col=col_idx)
        fig.update_yaxes(title_text="Count", row=1, col=col_idx)
        
        # Color channel statistics
        if has_color_stats:
            # Color channel means
            fig.add_trace(
                go.Bar(x=['Red', 'Green', 'Blue'],
                      y=[np.mean(stats['color_channels']['r_mean']),
                         np.mean(stats['color_channels']['g_mean']),
                         np.mean(stats['color_channels']['b_mean'])],
                      name='Mean Intensity', marker_color=['red', 'green', 'blue']),
                row=2, col=1
            )
            fig.update_xaxes(title_text="Channel", row=2, col=1)
            fig.update_yaxes(title_text="Mean Intensity (0-255)", row=2, col=1)
            
            # Color channel std dev
            fig.add_trace(
                go.Bar(x=['Red', 'Green', 'Blue'],
                      y=[np.mean(stats['color_channels']['r_std']),
                         np.mean(stats['color_channels']['g_std']),
                         np.mean(stats['color_channels']['b_std'])],
                      name='Std Dev', marker_color=['darkred', 'darkgreen', 'darkblue']),
                row=2, col=2
            )
            fig.update_xaxes(title_text="Channel", row=2, col=2)
            fig.update_yaxes(title_text="Std Deviation", row=2, col=2)
            
            # Saturation distribution
            if len(stats['saturation']) > 0:
                fig.add_trace(
                    go.Histogram(x=stats['saturation'], nbinsx=20, name='Saturation', marker_color='orange'),
                    row=2, col=3 if has_quality_stats else 3
                )
                sat_col = 3 if has_quality_stats else 3
                fig.update_xaxes(title_text="Saturation (0-255)", row=2, col=sat_col)
                fig.update_yaxes(title_text="Count", row=2, col=sat_col)
        
        # Quality metrics
        if has_quality_stats:
            row_idx = 3 if has_color_stats else 2
            # Sharpness distribution
            if len(stats['quality']['sharpness']) > 0:
                fig.add_trace(
                    go.Histogram(x=stats['quality']['sharpness'], nbinsx=20, name='Sharpness', marker_color='purple'),
                    row=row_idx, col=1
                )
                fig.update_xaxes(title_text="Sharpness (Laplacian Variance)", row=row_idx, col=1)
                fig.update_yaxes(title_text="Count", row=row_idx, col=1)
            
            # Contrast distribution
            fig.add_trace(
                go.Histogram(x=stats['quality']['contrast'], nbinsx=20, name='Contrast', marker_color='teal'),
                row=row_idx, col=2
            )
            fig.update_xaxes(title_text="Contrast (Std Dev)", row=row_idx, col=2)
            fig.update_yaxes(title_text="Count", row=row_idx, col=2)
        
        # Statistics table (comprehensive)
        stats_table = [
            ['Metric', 'Value'],
            ['Mean Width', f"{np.mean(stats['widths']):.1f} px"],
            ['Mean Height', f"{np.mean(stats['heights']):.1f} px"],
            ['Mean Aspect Ratio', f"{np.mean(stats['aspect_ratios']):.2f}"],
            ['Mean File Size', f"{np.mean(file_sizes_mb):.2f} MB"],
        ]
        
        if has_color_stats:
            stats_table.extend([
                ['Mean R Intensity', f"{np.mean(stats['color_channels']['r_mean']):.1f}"],
                ['Mean G Intensity', f"{np.mean(stats['color_channels']['g_mean']):.1f}"],
                ['Mean B Intensity', f"{np.mean(stats['color_channels']['b_mean']):.1f}"],
                ['Mean Saturation', f"{np.mean(stats['saturation']):.1f}"],
            ])
        
        if has_quality_stats:
            if len(stats['quality']['sharpness']) > 0:
                stats_table.append(['Mean Sharpness', f"{np.mean(stats['quality']['sharpness']):.2f}"])
            stats_table.append(['Mean Contrast', f"{np.mean(stats['quality']['contrast']):.2f}"])
        
        stats_table.extend([
            ['Min Width', f"{np.min(stats['widths'])} px"],
            ['Max Width', f"{np.max(stats['widths'])} px"],
            ['Min Height', f"{np.min(stats['heights'])} px"],
            ['Max Height', f"{np.max(stats['heights'])} px"]
        ])
        
        table_row = 3 if (has_color_stats and has_quality_stats) else (2 if has_color_stats else 2)
        table_col = 3 if (has_color_stats or has_quality_stats) else 2
        
        fig.add_trace(
            go.Table(
                header=dict(values=stats_table[0], fill_color='paleturquoise', align='left'),
                cells=dict(values=list(zip(*stats_table[1:])), fill_color='lavender', align='left')
            ),
            row=table_row, col=table_col
        )
        
        # Update layout
        height = 1000 if (has_color_stats and has_quality_stats) else (800 if has_color_stats else 800)
        fig.update_layout(
            title=f"📸 {title} - Enhanced Image Statistics (Sample: {len(stats['widths'])} images)",
            height=height,
            showlegend=False,
            template='plotly_white'
        )
        
        self._add_image_data_indicator(fig)
        return fig
    
    def _create_embedding_visualization(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Create t-SNE/UMAP visualization of image embeddings using pre-trained model"""
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            import torch
            from torchvision import transforms
            from torchvision.models import resnet18
            
            # Check if we have labels
            label_col = None
            for col in data.columns:
                if any(term in col.lower() for term in ['label', 'class', 'category', 'digit']):
                    label_col = col
                    break
            
            if not label_col:
                return self._create_classification_distribution(data, plot_spec, title)
            
            # Load pre-trained ResNet as feature extractor
            try:
                # Try new API first (torchvision >= 0.13)
                model = resnet18(weights='DEFAULT')
            except TypeError:
                # Fallback to old API
                model = resnet18(pretrained=True)
            model.eval()
            model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final FC layer
            
            # Image preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Extract embeddings
            image_cols = self._find_image_columns(data)
            if not image_cols or not PIL_AVAILABLE:
                return self._create_classification_distribution(data, plot_spec, title)
            
            embeddings = []
            labels = []
            sample_size = min(500, len(data))  # Limit for performance
            
            for idx, row in data.head(sample_size).iterrows():
                try:
                    image_data = str(row[image_cols[0]])
                    if image_data.startswith('data:image'):
                        base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                        img_bytes = base64.b64decode(base64_data)
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        img_tensor = transform(img).unsqueeze(0)
                        
                        with torch.no_grad():
                            embedding = model(img_tensor).squeeze().numpy().flatten()
                            embeddings.append(embedding)
                            labels.append(row[label_col])
                except Exception as e:
                    logger.debug(f"Could not process image {idx}: {e}")
                    continue
            
            if len(embeddings) < 10:
                return self._create_classification_distribution(data, plot_spec, title)
            
            embeddings = np.array(embeddings)
            
            # Use PCA first for dimensionality reduction, then t-SNE or UMAP
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(embeddings)
            
            # Try UMAP first (better for larger datasets), fallback to t-SNE
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings)-1))
                embeddings_2d = reducer.fit_transform(embeddings_pca)
                method_name = "UMAP"
            except ImportError:
                # Fallback to t-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
                embeddings_2d = tsne.fit_transform(embeddings_pca)
                method_name = "t-SNE"
            
            # Create visualization
            fig = px.scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                color=[str(l) for l in labels],
                title=f"📸 {title} - Embedding Space Visualization ({method_name})",
                labels={'x': f'{method_name} Dimension 1', 'y': f'{method_name} Dimension 2', 'color': label_col},
                hover_data={'label': labels}
            )
            
            fig.update_layout(height=600, template='plotly_white')
            self._add_image_data_indicator(fig)
            return fig
            
        except ImportError:
            logger.warning("torchvision not available for embedding visualization")
            return self._create_classification_distribution(data, plot_spec, title)
        except Exception as e:
            logger.error(f"Error creating embedding visualization: {e}")
            return self._create_classification_distribution(data, plot_spec, title)
    
    def _create_class_separability(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Assess class separability using embedding distances from pre-trained model"""
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            from scipy.spatial.distance import cdist
            import torch
            from torchvision import transforms
            from torchvision.models import resnet18
            
            label_col = None
            for col in data.columns:
                if any(term in col.lower() for term in ['label', 'class', 'category', 'digit']):
                    label_col = col
                    break
            
            if not label_col:
                return self._create_classification_distribution(data, plot_spec, title)
            
            # Load pre-trained ResNet as feature extractor
            try:
                model = resnet18(weights='DEFAULT')
            except TypeError:
                model = resnet18(pretrained=True)
            model.eval()
            model = torch.nn.Sequential(*list(model.children())[:-1])
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Extract embeddings
            image_cols = self._find_image_columns(data)
            if not image_cols or not PIL_AVAILABLE:
                # Fallback to metadata-based analysis
                class_counts = data[label_col].value_counts()
                imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Class Distribution', 'Class Balance Metrics'),
                    specs=[[{"type": "bar"}, {"type": "indicator"}]]
                )
                fig.add_trace(go.Bar(x=class_counts.index.astype(str), y=class_counts.values, name='Count'), row=1, col=1)
                fig.add_trace(go.Indicator(mode="gauge+number", value=min(imbalance_ratio, 10),
                    domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Imbalance Ratio"},
                    gauge={'axis': {'range': [None, 10]}, 'bar': {'color': "darkblue"},
                    'steps': [{'range': [0, 2], 'color': "lightgreen"}, {'range': [2, 5], 'color': "yellow"},
                    {'range': [5, 10], 'color': "red"}]}), row=1, col=2)
                fig.update_layout(title=f"📸 {title} - Class Separability Analysis", height=500, template='plotly_white')
                self._add_image_data_indicator(fig)
                return fig
            
            embeddings = []
            labels = []
            sample_size = min(300, len(data))
            
            for idx, row in data.head(sample_size).iterrows():
                try:
                    image_data = str(row[image_cols[0]])
                    if image_data.startswith('data:image'):
                        base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                        img_bytes = base64.b64decode(base64_data)
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        img_tensor = transform(img).unsqueeze(0)
                        
                        with torch.no_grad():
                            embedding = model(img_tensor).squeeze().numpy().flatten()
                            embeddings.append(embedding)
                            labels.append(row[label_col])
                except Exception as e:
                    logger.debug(f"Could not process image {idx}: {e}")
                    continue
            
            if len(embeddings) < 10:
                return self._create_classification_distribution(data, plot_spec, title)
            
            embeddings = np.array(embeddings)
            labels_array = np.array(labels)
            
            # Calculate class centroids and distances
            unique_labels = np.unique(labels_array)
            centroids = {}
            for label in unique_labels:
                mask = labels_array == label
                centroids[label] = np.mean(embeddings[mask], axis=0)
            
            # Calculate inter-class and intra-class distances
            inter_class_distances = []
            intra_class_distances = []
            
            for i, label1 in enumerate(unique_labels):
                for j, label2 in enumerate(unique_labels[i+1:], i+1):
                    dist = np.linalg.norm(centroids[label1] - centroids[label2])
                    inter_class_distances.append(dist)
                
                # Intra-class distances
                mask = labels_array == label1
                class_embeddings = embeddings[mask]
                if len(class_embeddings) > 1:
                    centroid = centroids[label1]
                    intra_dists = [np.linalg.norm(emb - centroid) for emb in class_embeddings]
                    intra_class_distances.extend(intra_dists)
            
            # Calculate silhouette score
            try:
                pca = PCA(n_components=50)
                embeddings_pca = pca.fit_transform(embeddings)
                silhouette = silhouette_score(embeddings_pca, labels_array)
            except:
                silhouette = 0.0
            
            # Calculate class imbalance ratio
            class_counts = pd.Series(labels_array).value_counts()
            max_count = class_counts.max()
            min_count = class_counts.min()
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            relative_counts = (class_counts / class_counts.sum() * 100).round(2)
            
            # Create visualization with enhanced imbalance metrics
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Class Distribution (Absolute)', 'Class Distribution (Relative %)', 
                              'Inter vs Intra-Class Distances', 'Class Imbalance Ratio',
                              'Class Separability Score', 'Distance Distribution'),
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                       [{"type": "indicator"}, {"type": "indicator"}, {"type": "histogram"}]]
            )
            
            # Class distribution (absolute counts)
            class_counts_sorted = class_counts.sort_index()
            fig.add_trace(
                go.Bar(x=class_counts_sorted.index.astype(str), y=class_counts_sorted.values, 
                      name='Count', marker_color='steelblue'),
                row=1, col=1
            )
            
            # Class distribution (relative percentages)
            relative_sorted = relative_counts.sort_index()
            fig.add_trace(
                go.Bar(x=relative_sorted.index.astype(str), y=relative_sorted.values,
                      name='Percentage', marker_color='lightblue', text=relative_sorted.values,
                      texttemplate='%{text:.1f}%', textposition='outside'),
                row=1, col=2
            )
            
            # Distance comparison
            fig.add_trace(
                go.Bar(x=['Inter-Class', 'Intra-Class'], 
                      y=[np.mean(inter_class_distances), np.mean(intra_class_distances)],
                      name='Distance', marker_color=['green', 'red']),
                row=1, col=3
            )
            
            # Class imbalance ratio indicator (enhanced)
            imbalance_display = min(imbalance_ratio, 20)  # Cap at 20 for display
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=imbalance_display,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Class Imbalance Ratio<br>(Max/Min)"},
                    delta={'reference': 1, 'position': "top", 'valueformat': '.2f'},
                    gauge={'axis': {'range': [None, 20]},
                           'bar': {'color': "darkred" if imbalance_ratio > 5 else ("orange" if imbalance_ratio > 2 else "darkgreen")},
                           'steps': [{'range': [0, 1.5], 'color': "lightgreen"},
                                    {'range': [1.5, 3], 'color': "yellow"},
                                    {'range': [3, 5], 'color': "orange"},
                                    {'range': [5, 20], 'color': "red"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 5}}
                ),
                row=2, col=1
            )
            
            # Separability indicator (silhouette score)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=max(0, min(1, silhouette)),  # Normalize to 0-1
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Separability Score<br>(Silhouette)"},
                    gauge={'axis': {'range': [None, 1]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 0.3], 'color': "red"},
                                    {'range': [0.3, 0.6], 'color': "yellow"},
                                    {'range': [0.6, 1], 'color': "lightgreen"}]}
                ),
                row=2, col=2
            )
            
            # Distance distribution
            fig.add_trace(
                go.Histogram(x=inter_class_distances + intra_class_distances,
                           name='Distances', nbinsx=20, marker_color='purple'),
                row=2, col=3
            )
            
            fig.update_xaxes(title_text="Class", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_xaxes(title_text="Class", row=1, col=2)
            fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
            fig.update_xaxes(title_text="Distance Type", row=1, col=3)
            fig.update_yaxes(title_text="Mean Distance", row=1, col=3)
            fig.update_xaxes(title_text="Distance", row=2, col=3)
            fig.update_yaxes(title_text="Frequency", row=2, col=3)
            
            # Add annotation with imbalance details
            imbalance_text = f"Max: {max_count}, Min: {min_count}, Ratio: {imbalance_ratio:.2f}"
            fig.add_annotation(
                text=f"<b>Imbalance Details:</b><br>{imbalance_text}",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=11)
            )
            
            fig.update_layout(
                title=f"📸 {title} - Class Separability & Imbalance Analysis (Embedding-based)",
                height=800,
                template='plotly_white'
            )
            
            self._add_image_data_indicator(fig)
            return fig
            
        except ImportError:
            logger.warning("torchvision not available for class separability")
            return self._create_classification_distribution(data, plot_spec, title)
        except Exception as e:
            logger.error(f"Error in class separability: {e}")
            return self._create_classification_distribution(data, plot_spec, title)
    
    def _create_duplicate_detection(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Detect duplicate/near-duplicate images using perceptual hashing"""
        try:
            import imagehash
            
            image_cols = self._find_image_columns(data)
            if not image_cols or not PIL_AVAILABLE:
                return self._create_classification_distribution(data, plot_spec, title)
            
            # Compute perceptual hashes
            hashes = {}
            sample_size = min(200, len(data))  # Limit for performance
            image_col = image_cols[0]
            
            for idx, row in data.head(sample_size).iterrows():
                try:
                    image_data = str(row[image_col])
                    if image_data.startswith('data:image'):
                        base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                        img_bytes = base64.b64decode(base64_data)
                        img = Image.open(io.BytesIO(img_bytes))
                        phash = imagehash.phash(img)
                        hashes[idx] = phash
                except Exception as e:
                    logger.debug(f"Could not hash image {idx}: {e}")
                    continue
            
            if len(hashes) < 2:
                return self._create_classification_distribution(data, plot_spec, title)
            
            # Find similar images (hamming distance < 5)
            similar_pairs = []
            hash_list = list(hashes.items())
            
            for i, (idx1, hash1) in enumerate(hash_list):
                for j, (idx2, hash2) in enumerate(hash_list[i+1:], i+1):
                    distance = hash1 - hash2
                    if distance < 5:  # Threshold for similarity
                        similar_pairs.append((idx1, idx2, distance))
            
            # Create visualization
            if similar_pairs:
                distances = [d for _, _, d in similar_pairs]
                fig = px.histogram(
                    x=distances,
                    title=f"📸 {title} - Duplicate Detection ({len(similar_pairs)} similar pairs found)",
                    labels={'x': 'Perceptual Hash Distance', 'y': 'Number of Pairs'},
                    nbins=10
                )
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"<b>No Duplicates Found</b><br>Analyzed {len(hashes)} images<br>All images appear unique",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(title=f"📸 {title} - Duplicate Detection")
            
            fig.update_layout(height=500, template='plotly_white')
            self._add_image_data_indicator(fig)
            return fig
            
        except ImportError:
            logger.warning("imagehash not available for duplicate detection")
            return self._create_classification_distribution(data, plot_spec, title)
        except Exception as e:
            logger.error(f"Error in duplicate detection: {e}")
            return self._create_classification_distribution(data, plot_spec, title)
    
    def _create_anomaly_detection(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Detect anomalies using pre-trained model embeddings and Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.decomposition import PCA
            import torch
            from torchvision import transforms
            from torchvision.models import resnet18
            
            image_cols = self._find_image_columns(data)
            if not image_cols or not PIL_AVAILABLE:
                return self._create_classification_distribution(data, plot_spec, title)
            
            # Load pre-trained ResNet as feature extractor
            try:
                model = resnet18(weights='DEFAULT')
            except TypeError:
                model = resnet18(pretrained=True)
            model.eval()
            model = torch.nn.Sequential(*list(model.children())[:-1])
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Extract embeddings
            embeddings = []
            indices = []
            sample_size = min(500, len(data))
            image_col = image_cols[0]
            
            for idx, row in data.head(sample_size).iterrows():
                try:
                    image_data = str(row[image_col])
                    if image_data.startswith('data:image'):
                        base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                        img_bytes = base64.b64decode(base64_data)
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        img_tensor = transform(img).unsqueeze(0)
                        
                        with torch.no_grad():
                            embedding = model(img_tensor).squeeze().numpy().flatten()
                            embeddings.append(embedding)
                            indices.append(idx)
                except Exception as e:
                    logger.debug(f"Could not process image {idx}: {e}")
                    continue
            
            if len(embeddings) < 10:
                return self._create_classification_distribution(data, plot_spec, title)
            
            embeddings = np.array(embeddings)
            
            # Reduce dimensionality for anomaly detection
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(embeddings)
            
            # Use Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(embeddings_pca)
            anomaly_scores = iso_forest.score_samples(embeddings_pca)
            
            # Separate normal and anomalous
            normal_indices = [indices[i] for i in range(len(indices)) if anomaly_labels[i] == 1]
            anomaly_indices = [indices[i] for i in range(len(indices)) if anomaly_labels[i] == -1]
            
            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Anomaly Score Distribution', 'Anomaly Detection Results',
                              'Anomaly Statistics', 'Score vs Index'),
                specs=[[{"type": "histogram"}, {"type": "bar"}],
                       [{"type": "indicator"}, {"type": "scatter"}]]
            )
            
            # Score distribution
            fig.add_trace(
                go.Histogram(x=anomaly_scores, nbinsx=30, name='Scores', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Anomaly counts
            normal_count = len(normal_indices)
            anomaly_count = len(anomaly_indices)
            fig.add_trace(
                go.Bar(x=['Normal', 'Anomaly'], y=[normal_count, anomaly_count],
                      marker_color=['green', 'red'], name='Count'),
                row=1, col=2
            )
            
            # Anomaly percentage indicator
            anomaly_pct = (anomaly_count / len(embeddings)) * 100
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=anomaly_pct,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Anomaly Percentage"},
                    gauge={'axis': {'range': [None, 20]},
                           'bar': {'color': "darkred"},
                           'steps': [{'range': [0, 5], 'color': "lightgreen"},
                                    {'range': [5, 10], 'color': "yellow"},
                                    {'range': [10, 20], 'color': "red"}]}
                ),
                row=2, col=1
            )
            
            # Score scatter
            colors = ['red' if label == -1 else 'green' for label in anomaly_labels]
            fig.add_trace(
                go.Scatter(x=list(range(len(embeddings))), y=anomaly_scores,
                          mode='markers', marker=dict(color=colors, size=5, opacity=0.6),
                          name='Scores'),
                row=2, col=2
            )
            
            fig.update_xaxes(title_text="Anomaly Score", row=1, col=1)
            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_xaxes(title_text="Category", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            fig.update_xaxes(title_text="Sample Index", row=2, col=2)
            fig.update_yaxes(title_text="Anomaly Score", row=2, col=2)
            
            fig.update_layout(
                title=f"📸 {title} - Anomaly Detection (Pre-trained Model)",
                height=700,
                template='plotly_white'
            )
            
            self._add_image_data_indicator(fig)
            return fig
            
        except ImportError:
            logger.warning("torchvision or sklearn not available for anomaly detection")
            return self._create_classification_distribution(data, plot_spec, title)
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return self._create_classification_distribution(data, plot_spec, title)
    
    def _create_class_similarity_matrix(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Create inter-class similarity matrix using ResNet18 embeddings - shows which classes are most similar"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import torch
            from torchvision import transforms
            from torchvision.models import resnet18
            
            label_col = None
            for col in data.columns:
                if any(term in col.lower() for term in ['label', 'class', 'category', 'digit']):
                    label_col = col
                    break
            
            if not label_col:
                return self._create_classification_distribution(data, plot_spec, title)
            
            # Load pre-trained ResNet as feature extractor
            try:
                model = resnet18(weights='DEFAULT')
            except TypeError:
                model = resnet18(pretrained=True)
            model.eval()
            model = torch.nn.Sequential(*list(model.children())[:-1])
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Extract embeddings for each class
            image_cols = self._find_image_columns(data)
            if not image_cols or not PIL_AVAILABLE:
                return self._create_classification_distribution(data, plot_spec, title)
            
            class_centroids = {}
            unique_labels = sorted(data[label_col].unique())
            
            print(f"Computing class centroids using ResNet18 for {len(unique_labels)} classes...")
            
            for label in unique_labels:
                class_data = data[data[label_col] == label].head(10)  # Sample up to 10 per class
                class_embeddings = []
                
                for idx, row in class_data.iterrows():
                    try:
                        image_data = str(row[image_cols[0]])
                        if image_data.startswith('data:image') or image_data.startswith('iVBORw0KGgo'):
                            base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                            img_bytes = base64.b64decode(base64_data)
                            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                            img_tensor = transform(img).unsqueeze(0)
                            
                            with torch.no_grad():
                                embedding = model(img_tensor).squeeze().numpy().flatten()
                                class_embeddings.append(embedding)
                    except Exception as e:
                        logger.debug(f"Could not process image for class {label}: {e}")
                        continue
                
                if len(class_embeddings) > 0:
                    class_centroids[label] = np.mean(class_embeddings, axis=0)
            
            if len(class_centroids) < 2:
                return self._create_classification_distribution(data, plot_spec, title)
            
            # Compute similarity matrix
            labels_list = sorted(class_centroids.keys())
            centroids_matrix = np.array([class_centroids[label] for label in labels_list])
            similarity_matrix = cosine_similarity(centroids_matrix)
            
            # Create visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Inter-Class Similarity Matrix (ResNet18)', 'Most Similar Class Pairs'),
                specs=[[{"type": "heatmap"}, {"type": "bar"}]]
            )
            
            # Heatmap
            fig.add_trace(
                go.Heatmap(
                    z=similarity_matrix,
                    x=[str(l) for l in labels_list],
                    y=[str(l) for l in labels_list],
                    colorscale='RdYlBu_r',
                    zmid=0.5,
                    text=similarity_matrix.round(3),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Cosine<br>Similarity", x=1.15)
                ),
                row=1, col=1
            )
            
            # Find most similar pairs (excluding diagonal)
            similar_pairs = []
            for i in range(len(labels_list)):
                for j in range(i+1, len(labels_list)):
                    similar_pairs.append({
                        'pair': f"{labels_list[i]} ↔ {labels_list[j]}",
                        'similarity': similarity_matrix[i, j]
                    })
            
            similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
            top_pairs = similar_pairs[:10]  # Top 10 most similar pairs
            
            fig.add_trace(
                go.Bar(
                    x=[p['pair'] for p in top_pairs],
                    y=[p['similarity'] for p in top_pairs],
                    marker_color='steelblue',
                    text=[f"{p['similarity']:.3f}" for p in top_pairs],
                    textposition='outside'
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Class", row=1, col=1)
            fig.update_yaxes(title_text="Class", row=1, col=1)
            fig.update_xaxes(title_text="Class Pair", row=1, col=2, tickangle=45)
            fig.update_yaxes(title_text="Cosine Similarity", row=1, col=2)
            
            fig.update_layout(
                title=f"📸 {title} - Inter-Class Similarity (ResNet18 Embeddings)",
                height=700,
                template='plotly_white'
            )
            
            self._add_image_data_indicator(fig)
            return fig
            
        except ImportError:
            logger.warning("torchvision not available for class similarity")
            return self._create_classification_distribution(data, plot_spec, title)
        except Exception as e:
            logger.error(f"Error in class similarity matrix: {e}")
            import traceback
            traceback.print_exc()
            return self._create_classification_distribution(data, plot_spec, title)
    
    def _create_nearest_neighbors(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Visualize nearest neighbors in ResNet18 embedding space - shows most similar images"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import torch
            from torchvision import transforms
            from torchvision.models import resnet18
            
            image_cols = self._find_image_columns(data)
            if not image_cols or not PIL_AVAILABLE:
                return self._create_classification_distribution(data, plot_spec, title)
            
            # Load pre-trained ResNet
            try:
                model = resnet18(weights='DEFAULT')
            except TypeError:
                model = resnet18(pretrained=True)
            model.eval()
            model = torch.nn.Sequential(*list(model.children())[:-1])
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Extract embeddings for a sample
            embeddings = []
            image_data_list = []
            labels_list = []
            sample_size = min(100, len(data))  # Limit for performance
            
            print(f"Computing nearest neighbors using ResNet18 for {sample_size} images...")
            
            for idx, row in data.head(sample_size).iterrows():
                try:
                    image_data = str(row[image_cols[0]])
                    if image_data.startswith('data:image') or image_data.startswith('iVBORw0KGgo'):
                        base64_data = image_data.split(',')[1] if ',' in image_data else image_data
                        img_bytes = base64.b64decode(base64_data)
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        img_tensor = transform(img).unsqueeze(0)
                        
                        with torch.no_grad():
                            embedding = model(img_tensor).squeeze().numpy().flatten()
                            embeddings.append(embedding)
                            image_data_list.append(image_data)
                            
                            # Get label if available
                            label_col = None
                            for col in data.columns:
                                if any(term in col.lower() for term in ['label', 'class', 'category', 'digit']):
                                    label_col = col
                                    break
                            labels_list.append(row[label_col] if label_col else 'Unknown')
                except Exception as e:
                    logger.debug(f"Could not process image {idx}: {e}")
                    continue
            
            if len(embeddings) < 5:
                return self._create_classification_distribution(data, plot_spec, title)
            
            embeddings = np.array(embeddings)
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find nearest neighbors for each image (excluding self)
            n_neighbors = min(5, len(embeddings) - 1)
            neighbor_info = []
            
            for i in range(len(embeddings)):
                similarities = similarity_matrix[i]
                similarities[i] = -1  # Exclude self
                nearest_indices = np.argsort(similarities)[-n_neighbors:][::-1]
                nearest_similarities = similarities[nearest_indices]
                
                neighbor_info.append({
                    'image_idx': i,
                    'label': labels_list[i],
                    'neighbors': [(nearest_indices[j], nearest_similarities[j]) for j in range(len(nearest_indices))]
                })
            
            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Average Similarity by Class', 'Nearest Neighbor Similarity Distribution',
                              'Cross-Class vs Same-Class Similarity', 'Top Similar Image Pairs'),
                specs=[[{"type": "bar"}, {"type": "histogram"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Average similarity by class
            label_col = None
            for col in data.columns:
                if any(term in col.lower() for term in ['label', 'class', 'category', 'digit']):
                    label_col = col
                    break
            
            if label_col:
                class_avg_similarity = {}
                for i, label in enumerate(labels_list):
                    if label not in class_avg_similarity:
                        class_avg_similarity[label] = []
                    # Average similarity to other images
                    avg_sim = np.mean([sim for j, sim in enumerate(similarity_matrix[i]) if i != j])
                    class_avg_similarity[label].append(avg_sim)
                
                class_means = {label: np.mean(sims) for label, sims in class_avg_similarity.items()}
                sorted_classes = sorted(class_means.items(), key=lambda x: x[1], reverse=True)
                
                fig.add_trace(
                    go.Bar(
                        x=[str(c[0]) for c in sorted_classes],
                        y=[c[1] for c in sorted_classes],
                        marker_color='steelblue',
                        name='Avg Similarity'
                    ),
                    row=1, col=1
                )
            
            # Similarity distribution
            all_similarities = []
            for i in range(len(similarity_matrix)):
                for j in range(i+1, len(similarity_matrix)):
                    all_similarities.append(similarity_matrix[i, j])
            
            fig.add_trace(
                go.Histogram(
                    x=all_similarities,
                    nbinsx=30,
                    marker_color='lightblue',
                    name='Similarity'
                ),
                row=1, col=2
            )
            
            # Cross-class vs same-class
            if label_col:
                same_class_sims = []
                cross_class_sims = []
                
                for i in range(len(similarity_matrix)):
                    for j in range(i+1, len(similarity_matrix)):
                        if labels_list[i] == labels_list[j]:
                            same_class_sims.append(similarity_matrix[i, j])
                        else:
                            cross_class_sims.append(similarity_matrix[i, j])
                
                fig.add_trace(
                    go.Bar(
                        x=['Same Class', 'Cross Class'],
                        y=[np.mean(same_class_sims) if same_class_sims else 0,
                           np.mean(cross_class_sims) if cross_class_sims else 0],
                        marker_color=['green', 'orange'],
                        name='Avg Similarity'
                    ),
                    row=2, col=1
                )
            
            # Top similar pairs
            all_pairs = []
            for i in range(len(similarity_matrix)):
                for j in range(i+1, len(similarity_matrix)):
                    all_pairs.append({
                        'pair': f"Img{i}↔Img{j}",
                        'similarity': similarity_matrix[i, j],
                        'labels': f"{labels_list[i]}↔{labels_list[j]}"
                    })
            
            all_pairs.sort(key=lambda x: x['similarity'], reverse=True)
            top_pairs = all_pairs[:10]
            
            fig.add_trace(
                go.Bar(
                    x=[p['labels'] for p in top_pairs],
                    y=[p['similarity'] for p in top_pairs],
                    marker_color='purple',
                    text=[f"{p['similarity']:.3f}" for p in top_pairs],
                    textposition='outside',
                    name='Similarity'
                ),
                row=2, col=2
            )
            
            fig.update_xaxes(title_text="Class", row=1, col=1)
            fig.update_yaxes(title_text="Avg Similarity", row=1, col=1)
            fig.update_xaxes(title_text="Cosine Similarity", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_xaxes(title_text="Category", row=2, col=1)
            fig.update_yaxes(title_text="Avg Similarity", row=2, col=1)
            fig.update_xaxes(title_text="Image Pair", row=2, col=2, tickangle=45)
            fig.update_yaxes(title_text="Similarity", row=2, col=2)
            
            fig.update_layout(
                title=f"📸 {title} - Nearest Neighbors Analysis (ResNet18 Embeddings)",
                height=900,
                template='plotly_white'
            )
            
            self._add_image_data_indicator(fig)
            return fig
            
        except ImportError:
            logger.warning("torchvision not available for nearest neighbors")
            return self._create_classification_distribution(data, plot_spec, title)
        except Exception as e:
            logger.error(f"Error in nearest neighbors: {e}")
            import traceback
            traceback.print_exc()
            return self._create_classification_distribution(data, plot_spec, title)
    
    def _create_train_test_distribution(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Create train vs test split distribution visualization - data analysis only"""
        # Find split column
        split_col = None
        for col in data.columns:
            if col.lower() == 'split':
                split_col = col
                break
        
        if not split_col:
            # Fallback to classification distribution
            return self._create_classification_distribution(data, plot_spec, title)
        
        # Get split counts
        split_counts = data[split_col].value_counts()
        total = len(data)
        
        # Check if we have label column for stacked view
        label_col = None
        for col in data.columns:
            col_lower = col.lower()
            if col_lower in ['label', 'digit']:
                label_col = col
                break
        
        if label_col:
            # Create stacked bar chart showing class distribution within each split
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Train vs Test Split (Counts)', 'Class Distribution by Split'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Split counts
            fig.add_trace(
                go.Bar(
                    x=split_counts.index.astype(str),
                    y=split_counts.values,
                    name='Count',
                    marker_color=['steelblue', 'lightblue'],
                    text=split_counts.values,
                    textposition='outside',
                    hovertemplate='Split: %{x}<br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
                    customdata=(split_counts.values / total * 100)
                ),
                row=1, col=1
            )
            
            # Class distribution by split (stacked)
            unique_splits = sorted(data[split_col].unique())
            unique_labels = sorted(data[label_col].unique())
            
            for split in unique_splits:
                split_data = data[data[split_col] == split]
                class_counts = split_data[label_col].value_counts().reindex(unique_labels, fill_value=0)
                fig.add_trace(
                    go.Bar(
                        x=class_counts.index.astype(str),
                        y=class_counts.values,
                        name=f'{split}',
                        hovertemplate='Class: %{x}<br>Count: %{y}<extra></extra>'
                    ),
                    row=1, col=2
                )
            
            fig.update_xaxes(title_text="Split", row=1, col=1)
            fig.update_yaxes(title_text="Number of Images", row=1, col=1)
            fig.update_xaxes(title_text="Class/Label", row=1, col=2)
            fig.update_yaxes(title_text="Number of Images", row=1, col=2)
            
        else:
            # Simple split counts only
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=split_counts.index.astype(str),
                    y=split_counts.values,
                    marker_color=['steelblue', 'lightblue'],
                    text=split_counts.values,
                    textposition='outside',
                    hovertemplate='Split: %{x}<br>Count: %{y}<br>Percentage: %{customdata:.1f}%<extra></extra>',
                    customdata=(split_counts.values / total * 100)
                )
            )
            fig.update_xaxes(title_text="Split")
            fig.update_yaxes(title_text="Number of Images")
        
        fig.update_layout(
            title=f"📸 {title} - Train/Test Split Distribution (Data Analysis)",
            height=500,
            template='plotly_white',
            barmode='group' if label_col else 'group'
        )
        
        # Add annotation
        train_pct = (split_counts.get('train', 0) / total * 100) if 'train' in split_counts.index else 0
        test_pct = (split_counts.get('test', 0) / total * 100) if 'test' in split_counts.index else 0
        fig.add_annotation(
            text=f"<b>📸 DATA EXPLORATION</b><br>Train: {split_counts.get('train', 0)} ({train_pct:.1f}%)<br>Test: {split_counts.get('test', 0)} ({test_pct:.1f}%)<br><i>Dataset split analysis</i>",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(52, 152, 219, 0.15)",
            bordercolor="rgba(52, 152, 219, 0.5)",
            borderwidth=2,
            font=dict(size=11)
        )
        
        self._add_image_data_indicator(fig)
        return fig
    
    def _create_dataset_overview(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Create dataset overview summary card - comprehensive dataset information"""
        # Find key columns
        label_col = None
        split_col = None
        image_cols = self._find_image_columns(data)
        
        for col in data.columns:
            col_lower = col.lower()
            if col_lower in ['label', 'digit']:
                label_col = col
            elif col_lower == 'split':
                split_col = col
        
        # Calculate statistics
        total_images = len(data)
        num_classes = len(data[label_col].unique()) if label_col else 'N/A'
        classes = sorted(data[label_col].unique().tolist()) if label_col else []
        
        # Split information
        split_info = {}
        if split_col:
            split_counts = data[split_col].value_counts()
            split_info = split_counts.to_dict()
        
        # Column information
        column_info = []
        for col in data.columns:
            dtype = str(data[col].dtype)
            non_null = data[col].notna().sum()
            column_info.append({
                'Column': col,
                'Type': dtype,
                'Non-Null': non_null,
                'Null': total_images - non_null
            })
        
        # Create table visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Dataset Summary', 'Split Distribution', 'Column Information', 'Class List'),
            specs=[[{"type": "table"}, {"type": "bar"}],
                   [{"type": "table"}, {"type": "table"}]]
        )
        
        # Dataset summary table
        summary_data = [
            ['Total Images', str(total_images)],
            ['Number of Classes', str(num_classes)],
            ['Image Columns', str(len(image_cols))],
            ['Total Columns', str(len(data.columns))],
        ]
        if split_col:
            summary_data.append(['Has Split Info', 'Yes'])
            summary_data.append(['Train Count', str(split_info.get('train', 'N/A'))])
            summary_data.append(['Test Count', str(split_info.get('test', 'N/A'))])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'], fill_color='paleturquoise', align='left'),
                cells=dict(values=list(zip(*summary_data)), fill_color='lavender', align='left')
            ),
            row=1, col=1
        )
        
        # Split distribution bar
        if split_col:
            split_counts = data[split_col].value_counts()
            fig.add_trace(
                go.Bar(
                    x=split_counts.index.astype(str),
                    y=split_counts.values,
                    marker_color=['steelblue', 'lightblue'],
                    name='Count'
                ),
                row=1, col=2
            )
            fig.update_xaxes(title_text="Split", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=2)
        else:
            fig.add_trace(
                go.Bar(x=['Total'], y=[total_images], marker_color='steelblue'),
                row=1, col=2
            )
        
        # Column information table
        col_table_data = list(zip(*[[c['Column'], c['Type'], str(c['Non-Null']), str(c['Null'])] for c in column_info[:10]]))
        fig.add_trace(
            go.Table(
                header=dict(values=['Column', 'Type', 'Non-Null', 'Null'], fill_color='paleturquoise', align='left'),
                cells=dict(values=col_table_data, fill_color='lavender', align='left')
            ),
            row=2, col=1
        )
        
        # Class list table
        if classes:
            class_data = [['Class', 'Count']]
            if label_col:
                class_counts = data[label_col].value_counts().sort_index()
                for cls in classes[:20]:  # Limit to 20 classes
                    class_data.append([str(cls), str(class_counts.get(cls, 0))])
            else:
                for cls in classes[:20]:
                    class_data.append([str(cls), 'N/A'])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=class_data[0], fill_color='paleturquoise', align='left'),
                    cells=dict(values=list(zip(*class_data[1:])), fill_color='lavender', align='left')
                ),
                row=2, col=2
            )
        else:
            fig.add_trace(
                go.Table(
                    header=dict(values=['Info'], fill_color='paleturquoise', align='left'),
                    cells=dict(values=[['No class information available']], fill_color='lavender', align='left')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"📸 {title} - Dataset Overview (Data Analysis)",
            height=800,
            template='plotly_white'
        )
        
        self._add_image_data_indicator(fig)
        return fig
    
    def _create_class_balance_analysis(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Create class balance/imbalance analysis - data exploration only"""
        # Find label column
        label_col = None
        for col in data.columns:
            col_lower = col.lower()
            if col_lower in ['label', 'digit']:
                label_col = col
                break
        
        if not label_col:
            return self._create_classification_distribution(data, plot_spec, title)
        
        class_counts = data[label_col].value_counts().sort_index()
        total = len(data)
        percentages = (class_counts / total * 100).round(2)
        
        # Calculate imbalance metrics
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Class Distribution (Counts)', 'Class Distribution (Percentages)',
                          'Imbalance Ratio', 'Balance Status'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Counts bar chart
        fig.add_trace(
            go.Bar(
                x=class_counts.index.astype(str),
                y=class_counts.values,
                name='Count',
                marker_color='steelblue',
                text=class_counts.values,
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Percentages bar chart
        fig.add_trace(
            go.Bar(
                x=percentages.index.astype(str),
                y=percentages.values,
                name='Percentage',
                marker_color='lightblue',
                text=[f'{p:.1f}%' for p in percentages.values],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # Imbalance ratio indicator
        imbalance_display = min(imbalance_ratio, 10)  # Cap at 10 for display
        balance_color = 'green' if imbalance_ratio < 1.5 else ('orange' if imbalance_ratio < 3 else 'red')
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=imbalance_display,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Imbalance Ratio<br>(Max/Min)"},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': balance_color},
                    'steps': [
                        {'range': [0, 1.5], 'color': "lightgreen"},
                        {'range': [1.5, 3], 'color': "yellow"},
                        {'range': [3, 10], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 3
                    }
                }
            ),
            row=2, col=1
        )
        
        # Balance status indicator
        if imbalance_ratio < 1.5:
            status = "Well Balanced"
            status_color = "green"
        elif imbalance_ratio < 3:
            status = "Moderately Imbalanced"
            status_color = "orange"
        else:
            status = "Highly Imbalanced"
            status_color = "red"
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=imbalance_ratio,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Balance Status<br>{status}"},
                delta={'reference': 1, 'position': "top"},
                number={'font': {'color': status_color}}
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Class", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Class", row=1, col=2)
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
        
        fig.update_layout(
            title=f"📸 {title} - Class Balance Analysis (Data Exploration)",
            height=700,
            template='plotly_white',
            showlegend=False
        )
        
        # Add annotation with details
        fig.add_annotation(
            text=f"<b>📸 DATA EXPLORATION</b><br>Max: {max_count}, Min: {min_count}<br>Ratio: {imbalance_ratio:.2f}<br>Total: {total} images<br><i>Analyzing data balance, not model performance</i>",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(52, 152, 219, 0.15)",
            bordercolor="rgba(52, 152, 219, 0.5)",
            borderwidth=2,
            font=dict(size=11)
        )
        
        self._add_image_data_indicator(fig)
        return fig
    
    def _create_image_grid_by_class(self, data: pd.DataFrame, plot_spec: Dict[str, Any], title: str):
        """Create image grid organized by class - enhanced version of image grid"""
        image_cols = self._find_image_columns(data)
        label_col = None
        
        for col in data.columns:
            col_lower = col.lower()
            if col_lower in ['label', 'digit']:
                label_col = col
                break
        
        if not image_cols:
            return self._create_classification_distribution(data, plot_spec, title)
        
        image_col = image_cols[0]
        
        # Group by label if available
        if label_col:
            grouped = data.groupby(label_col)
            html_images = []
            
            # Get 2-3 samples per class
            for label, group in grouped:
                for idx, (_, row) in enumerate(group.head(3).iterrows()):
                    image_data = str(row[image_col])
                    html_images.append({
                        'image': image_data,
                        'label': row[label_col],
                        'id': f"img_{len(html_images)}"
                    })
        else:
            # Just take first N images
            html_images = []
            for idx, (_, row) in enumerate(data.head(20).iterrows()):
                image_data = str(row[image_col])
                html_images.append({
                    'image': image_data,
                    'label': None,
                    'id': f"img_{idx}"
                })
        
        # Create HTML-based image grid organized by class
        n_cols = 5
        grid_html = f"""
        <div style="padding: 20px; background: white; font-family: Arial, sans-serif;">
            <h3 style="text-align: center; margin-bottom: 20px; color: #2c3e50;">{title}</h3>
            <p style="text-align: center; color: #7f8c8d; margin-bottom: 20px;">📸 Image Grid by Class - Showing sample images from dataset</p>
            <div style="display: grid; grid-template-columns: repeat({n_cols}, 1fr); gap: 15px; max-width: 1200px; margin: 0 auto;">
        """
        
        for img_info in html_images:
            label_text = f"<div style='text-align: center; font-weight: bold; margin-top: 8px; color: #34495e; font-size: 14px;'>Class: {img_info['label']}</div>" if img_info['label'] is not None else ""
            grid_html += f"""
                <div style="border: 2px solid #bdc3c7; border-radius: 8px; padding: 10px; background: #ecf0f1; text-align: center; transition: transform 0.2s;">
                    <img src="{img_info['image']}" 
                         style="max-width: 100%; height: auto; max-height: 120px; border-radius: 4px; display: block; margin: 0 auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" 
                         alt="Image {img_info['id']}"
                         onerror="this.style.display='none'; this.parentElement.innerHTML='<div style=\\'padding: 20px; color: #e74c3c;\\'>⚠️ Image failed to load</div>';">
                    {label_text}
                </div>
            """
        
        grid_html += """
            </div>
            <div style="text-align: center; margin-top: 20px; color: #95a5a6; font-size: 12px;">
                <p>💡 This is a <strong>DATA EXPLORATION</strong> visualization showing actual images from your dataset</p>
            </div>
        </div>
        """
        
        return {'type': 'html_grid', 'html': grid_html, 'title': title}
    
    def get_supported_queries(self) -> list:
        """Return examples of supported query types"""
        return [
            "Show image classification distribution",
            "Show class distribution",
            "Show label distribution",
            "Show train test distribution",
            "Show dataset overview",
            "Show class balance analysis",
            "Show image statistics",
            "Show sample images by class",
            "Create confusion matrix for image predictions",
            "Display image grid showing actual images",
            "Visualize image embeddings (t-SNE/UMAP) - Uses ResNet18 pre-trained model",
            "Analyze class separability - Uses ResNet18 pre-trained model",
            "Detect duplicate images",
            "Detect anomalous/outlier images - Uses ResNet18 pre-trained model",
            "Show image count by class",
            "Plot image metadata correlation",
            "Generate histogram of image labels",
            "Show classification results",
            "Create heatmap of image features",
            "Display images from the dataset",
            "Show embedding visualization - Uses ResNet18 (ImageNet pre-trained)",
            "Analyze class separability with embeddings - Uses ResNet18",
            "Show class similarity matrix - Uses ResNet18 to find most similar classes",
            "Analyze nearest neighbors - Uses ResNet18 to find most similar images",
            "Show inter-class similarity - ResNet18 embedding-based heatmap"
        ]

