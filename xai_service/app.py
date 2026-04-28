from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import lime
import lime.lime_tabular
import base64
import io
import json
from datetime import datetime
import hashlib
import warnings
import logging
import re
from typing import List, Dict, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from lime.lime_text import LimeTextExplainer
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
SHARED_DATA_DIR = '/app/shared_data'
UPLOAD_FOLDER = os.path.join(SHARED_DATA_DIR, 'uploads')
MODELS_FOLDER = os.path.join(SHARED_DATA_DIR, 'models')
RESULTS_FOLDER = os.path.join(SHARED_DATA_DIR, 'results')
PLOTS_FOLDER = os.path.join(SHARED_DATA_DIR, 'plots')
AI_OUTPUTS_SERVICE_URL = os.environ.get('AI_OUTPUTS_SERVICE_URL', 'http://ai_outputs:8001')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Global variables to store data and model - make them more persistent
import threading
_data_store_lock = threading.Lock()
data_store = {}
model_store = {}

# Custom JSON encoder to handle timestamps and numpy objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'isoformat'):  # Handle datetime/timestamp objects
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):  # Handle numpy dtypes
            return str(obj)
        elif hasattr(obj, 'name'):  # Handle pandas columns
            return str(obj)
        elif hasattr(obj, 'dtype') and hasattr(obj.dtype, 'name'):  # Handle pandas dtypes more specifically
            return str(obj.dtype.name)
        elif hasattr(obj, 'index'):  # Handle pandas Index objects
            return obj.tolist()
        elif hasattr(obj, 'values'):  # Handle pandas Series
            return obj.values.tolist()
        return super().default(obj)

def preprocess_timeseries_data(df):
    """Preprocess timeseries data: sort by date, extract features, create lags and rolling stats.
    Returns (processed_df, target_column)."""
    df = df.copy()

    # Detect date column
    date_col = None
    for col in df.columns:
        if col.lower() in ('date', 'datetime', 'timestamp', 'time'):
            date_col = col
            break
    if date_col is None:
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head(20))
                date_col = col
                break
            except Exception:
                continue

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col).reset_index(drop=True)
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek

    # Detect target column
    target_column = None
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for candidate in ('close', 'Close', 'price', 'Price', 'value', 'Value',
                       'target', 'Target', 'y'):
        if candidate in numeric_cols:
            target_column = candidate
            break
    if target_column is None and numeric_cols:
        target_column = numeric_cols[-1]

    # Create lag and rolling features for the target
    if target_column:
        for lag in (1, 3, 7):
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        for window in (7, 14, 30):
            df[f'{target_column}_roll_mean_{window}'] = df[target_column].rolling(window).mean()
            df[f'{target_column}_roll_std_{window}'] = df[target_column].rolling(window).std()
        df = df.dropna().reset_index(drop=True)

    return df, target_column


def load_news_sentiment_data(file_path):
    """Load and process news sentiment JSON data extracting title, asset (symbol), and sentiment."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        records = []
        for date, companies in data.items():
            for company, articles in companies.items():
                for article in articles:
                    try:
                        title = article.get('title', '')
                        sentiment = article.get('sentiment', 0.0)
                        symbols = article.get('symbols', [])
                        asset = symbols[0] if symbols else company
                        records.append({
                            'title': title,
                            'asset': asset,
                            'sentiment': sentiment,
                        })
                    except Exception as e:
                        print(f"Warning: Skipping article due to error: {e}")
                        continue

        df = pd.DataFrame(records)
        df['title'] = df['title'].str.replace(r'[^\w\s\.,!?-]', ' ', regex=True)
        df['title'] = df['title'].str.replace(r'\s+', ' ', regex=True)
        df['title'] = df['title'].str.strip()

        def categorize_sentiment(score):
            if score > 0.1:
                return 'positive'
            elif score < -0.1:
                return 'negative'
            return 'neutral'

        df['sentiment_label'] = df['sentiment'].apply(categorize_sentiment)
        df = df.dropna(subset=['title'])
        df = df[df['title'].str.len() > 5]

        print(f"Loaded {len(df)} news articles with title, asset, and sentiment")
        print(f"Sentiment distribution: {df['sentiment_label'].value_counts().to_dict()}")
        print(f"Assets analyzed: {df['asset'].nunique()}")
        return df

    except Exception as e:
        raise Exception(f"Error loading news sentiment data: {str(e)}")


def load_data(file_path):
    """Load data from various file formats"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            # Check if it's news sentiment data
            if 'news_sentiment' in file_path or 'sentiment' in file_path:
                return load_news_sentiment_data(file_path)
            else:
                return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def detect_data_type(df):
    """Automatically detect data type (text, image, timeseries, or tabular)"""
    data_type_info = {
        'type': 'tabular',
        'confidence': 0.0,
        'features': {},
        'preprocessing_needed': []
    }
    
    # Check for image data first (highest priority)
    image_columns = []
    for col in df.columns:
        col_lower = col.lower()
        # Check column name
        if any(term in col_lower for term in ['image', 'img', 'picture', 'photo', 'file_path', 'path']):
            image_columns.append(col)
            continue
        
        # Check if column contains base64 image data
        if df[col].dtype == 'object':
            sample_values = df[col].dropna().head(5)
            if len(sample_values) > 0:
                first_val = str(sample_values.iloc[0])
                # Check for base64 image data
                if (first_val.startswith('data:image') or 
                    (len(first_val) > 100 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in first_val[:100]))):
                    image_columns.append(col)
                    continue
                # Check for image file paths
                if any(ext in first_val.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']):
                    image_columns.append(col)
                    continue
    
    if len(image_columns) > 0:
        data_type_info['type'] = 'image'
        data_type_info['confidence'] = 0.9
        data_type_info['features']['image_columns'] = image_columns
        data_type_info['preprocessing_needed'].extend(['image_loading', 'image_preprocessing', 'feature_extraction'])
        return data_type_info
    
    # Check for time series data
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
    
    # Check if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        datetime_cols.append('__index__')
    
    if len(datetime_cols) > 0:
        data_type_info['type'] = 'timeseries'
        data_type_info['confidence'] = 0.85
        data_type_info['features']['datetime_columns'] = datetime_cols
        data_type_info['preprocessing_needed'].extend(['datetime_parsing', 'time_series_features', 'resampling'])
        return data_type_info
    
    # Check for text data
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object' and col not in image_columns:
            # Check if column contains text (not categorical)
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                avg_length = sample_values.astype(str).str.len().mean()
                if avg_length > 20:  # Likely text if average length > 20 chars
                    text_columns.append(col)
    
    if len(text_columns) > 0 and len(text_columns) == len(df.select_dtypes(include=['object']).columns):
        data_type_info['type'] = 'text'
        data_type_info['confidence'] = 0.8
        data_type_info['features']['text_columns'] = text_columns
        data_type_info['preprocessing_needed'].extend(['text_cleaning', 'tokenization', 'feature_extraction'])
        return data_type_info
    
    # If not text, image, or timeseries, it's tabular
    if data_type_info['type'] == 'tabular':
        data_type_info['confidence'] = 0.9
        data_type_info['preprocessing_needed'].extend(['handle_missing', 'encode_categorical', 'normalize_numeric'])
    
    return data_type_info



def preprocess_text_data(df, target_column=None, text_column=None):
    """Preprocess text data"""
    df_processed = df.copy()
    
    # Find text column if not specified
    if text_column is None:
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(10)
                avg_length = sample_values.astype(str).str.len().mean()
                if avg_length > 20:
                    text_columns.append(col)
        text_column = text_columns[0] if text_columns else None
    
    if text_column:
        # Clean text
        df_processed[f'{text_column}_cleaned'] = df_processed[text_column].astype(str)
        df_processed[f'{text_column}_cleaned'] = df_processed[f'{text_column}_cleaned'].str.lower()
        df_processed[f'{text_column}_cleaned'] = df_processed[f'{text_column}_cleaned'].str.replace(r'[^\w\s]', ' ', regex=True)
        df_processed[f'{text_column}_cleaned'] = df_processed[f'{text_column}_cleaned'].str.replace(r'\s+', ' ', regex=True)
        df_processed[f'{text_column}_cleaned'] = df_processed[f'{text_column}_cleaned'].str.strip()
        
        # Create text features
        df_processed[f'{text_column}_length'] = df_processed[f'{text_column}_cleaned'].str.len()
        df_processed[f'{text_column}_word_count'] = df_processed[f'{text_column}_cleaned'].str.split().str.len()
        df_processed[f'{text_column}_avg_word_length'] = df_processed[f'{text_column}_cleaned'].str.split().apply(
            lambda x: np.mean([len(word) for word in x]) if x else 0
        )
    
    # Find target column if not specified
    if target_column is None:
        # Look for sentiment or label columns
        possible_targets = ['sentiment', 'label', 'target', 'class', 'category']
        for col in df.columns:
            if any(target in col.lower() for target in possible_targets):
                target_column = col
                break
        
        # If no target found, use first categorical column
        if target_column is None:
            categorical_columns = df.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                target_column = categorical_columns[0]
    
    return df_processed, target_column, text_column

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'xai_service'})

@app.route('/ingest', methods=['POST'])
def ingest_data():
    """Enhanced data ingestion with automatic data type detection"""
    try:
        data = request.json
        file_path = data.get('file_path')
        user_id = data.get('user_id')
        data_type = data.get('data_type', 'auto')  # auto, timeseries, text, tabular
        
        if not file_path or not user_id:
            return jsonify({'error': 'Missing file_path or user_id'}), 400
        
        print(f"Ingesting data for user {user_id}: {file_path}")
        
        # Load data
        df = load_data(file_path)
        print(f"Loaded data shape: {df.shape}")
        
        # Detect data type if auto
        if data_type == 'auto':
            data_type_info = detect_data_type(df)
            data_type = data_type_info['type']
            print(f"Detected data type: {data_type} (confidence: {data_type_info['confidence']})")
        else:
            data_type_info = detect_data_type(df)
        
        # Preprocess based on data type
        if data_type == 'timeseries':
            df_processed, target_column = preprocess_timeseries_data(df)
            # Rename columns for model compatibility
            rename_map = {'year': 'Year', 'month': 'Month', 'day': 'Day', 'day_of_week': 'DayOfWeek'}
            for old, new in rename_map.items():
                if old in df_processed.columns:
                    df_processed.rename(columns={old: new}, inplace=True)
            preprocessing_info = {
                'data_type': 'timeseries',
                'target_column': target_column,
                'preprocessing_steps': ['sort_by_date', 'create_lags', 'rolling_features', 'rename_columns_for_model']
            }
        elif data_type == 'text':
            df_processed, target_column, text_column = preprocess_text_data(df)
            preprocessing_info = {
                'data_type': 'text',
                'target_column': target_column,
                'text_column': text_column,
                'preprocessing_steps': ['text_cleaning', 'feature_extraction']
            }
        elif data_type == 'image':
            # For image data, keep original structure but add metadata
            df_processed = df.copy()
            image_columns = data_type_info.get('features', {}).get('image_columns', [])
            preprocessing_info = {
                'data_type': 'image',
                'image_columns': image_columns,
                'preprocessing_steps': ['image_validation', 'metadata_extraction']
            }
        else:  # tabular
            df_processed = df.copy()
            preprocessing_info = {
                'data_type': 'tabular',
                'preprocessing_steps': ['basic_validation']
            }
        
        # Store processed data
        print(f"DEBUG: Storing data for user {user_id} with shape {df_processed.shape}")
        print(f"DEBUG: Data type: {data_type}, Columns: {list(df_processed.columns)[:5]}")
        try:
            with _data_store_lock:
                data_store[user_id] = {
                    'data': df_processed,
                    'original_data': df,
                    'file_path': file_path,
                    'data_type': data_type,
                    'preprocessing_info': preprocessing_info,
                    'ingested_at': datetime.now().isoformat()
                }
                print(f"DEBUG: Data store now has {len(data_store)} users: {list(data_store.keys())}")
                print(f"DEBUG: Successfully stored data for user {user_id}")
        except Exception as store_error:
            print(f"ERROR: Failed to store data in data_store: {store_error}")
            import traceback
            traceback.print_exc()
            raise
        
        # Generate data summary with proper serialization
        data_summary = {
            'shape': df_processed.shape,
            'columns': df_processed.columns.tolist(),
            'data_types': {str(k): str(v) for k, v in df_processed.dtypes.to_dict().items()},
            'missing_values': {str(k): int(v) for k, v in df_processed.isnull().sum().to_dict().items()},
            'numeric_columns': df_processed.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df_processed.select_dtypes(include=['object']).columns.tolist(),
            'data_type': data_type,
            'preprocessing_info': preprocessing_info
        }
        
        # Send data to AI outputs service for storage
        try:
            import requests
            requests.post(f"{AI_OUTPUTS_SERVICE_URL}/store-data", json={
                'user_id': user_id,
                'data_info': data_summary
            })
        except Exception as e:
            print(f"Failed to send data to AI outputs service: {e}")
        
        print(f"Data ingestion completed. Processed shape: {df_processed.shape}")
        
        return jsonify({
            'message': 'Data ingested successfully',
            'data_summary': data_summary,
            'data_type': data_type
        })
        
    except Exception as e:
        print(f"Error in ingest_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/samples/<user_id>', methods=['GET'])
def get_samples(user_id):
    """Return N rows from the user's ingested dataset for per-sample XAI display."""
    try:
        n = min(int(request.args.get('n', 50)), 500)

        with _data_store_lock:
            if user_id not in data_store:
                return jsonify({'error': 'No data found. Please upload/select a dataset first.'}), 404
            user_data = data_store[user_id]

        df = user_data['data']
        data_type = user_data.get('data_type', 'tabular')

        # Auto-detect text and sentiment columns
        text_col = None
        sent_col = None
        for col in df.columns:
            cl = col.lower()
            if text_col is None and any(t in cl for t in ['text', 'title', 'content', 'sentence', 'review', 'headline']):
                text_col = col
            if sent_col is None and any(t in cl for t in ['sentiment', 'finbert_sentiment', 'label', 'target', 'class']):
                sent_col = col

        # For image data, detect image/path columns
        image_col = None
        for col in df.columns:
            cl = col.lower()
            if any(t in cl for t in ['image', 'img', 'file_path', 'path', 'photo', 'filename']):
                image_col = col
                break

        # Build samples list
        samples = []
        for i, (_, row) in enumerate(df.head(n).iterrows()):
            sample = {'index': i}
            if text_col:
                sample['text'] = str(row[text_col])
            if sent_col:
                sample['sentiment'] = str(row[sent_col])
            if image_col:
                sample['image_path'] = str(row[image_col])
            # Include asset if present
            if 'asset' in df.columns:
                sample['asset'] = str(row['asset'])
            samples.append(sample)

        return jsonify({
            'samples': samples,
            'total': len(df),
            'returned': len(samples),
            'data_type': data_type,
            'columns': {
                'text': text_col,
                'sentiment': sent_col,
                'image': image_col,
                'all': df.columns.tolist(),
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/data-statistics', methods=['POST'])
def data_statistics():
    """Generate and return basic overview plot automatically. Additional plots can be requested via plot selection."""
    try:
        data = request.json
        user_id = data.get('user_id')
        plot_types = data.get('plot_types', [])  # Optional: list of specific plot types to generate
        
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        
        print(f"=== DATA STATISTICS CALLED ===")
        print(f"User ID: {user_id}")
        print(f"Requested plot types: {plot_types if plot_types else 'AUTO (overview only)'}")
        print(f"Available users in data_store: {list(data_store.keys())}")
        
        with _data_store_lock:
            if user_id not in data_store:
                print(f"ERROR: User {user_id} not found in data_store")
                return jsonify({'error': 'No data found for user. Please upload data first.'}), 400
            
            user_data = data_store[user_id]
            df = user_data['data'].copy()  # Make a copy to avoid issues
            data_type = user_data.get('data_type', 'tabular')
        
        print(f"Data type: {data_type}")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Initialize images list
        images = []
        
        # If no specific plot types requested, only generate basic overview
        generate_all = len(plot_types) == 0
        
        if data_type == 'image':
            print("DEBUG: Processing as IMAGE data - using ImagePlotGenerator for enhanced plots")
            try:
                import os
                from datetime import datetime
                import base64
                import io
                
                # Import ImagePlotGenerator
                from plot_generators.registry import PlotGeneratorRegistry
                generator = PlotGeneratorRegistry.get_generator('image')
                
                user_results_dir = f"/app/shared_data/results/{user_id}"
                os.makedirs(user_results_dir, exist_ok=True)
                
                # Clear old image statistics files
                import glob
                old_files = glob.glob(os.path.join(user_results_dir, "image_*.png"))
                for old_file in old_files:
                    try:
                        os.remove(old_file)
                    except:
                        pass
                
                base64_images = []
                
                # 1. Enhanced Image Statistics (with color channels & quality metrics)
                try:
                    plot_spec = {'plot_type': 'image_statistics', 'title': '📸 Image Statistics'}
                    fig = generator._create_image_statistics(df, plot_spec, 'Image Statistics')
                    # Note: _create_image_statistics already calls _add_image_data_indicator internally at line 970
                    
                    # Convert Plotly figure to PNG using kaleido
                    try:
                        img_bytes = fig.to_image(format="png", width=1400, height=1000, scale=2)
                        base64_images.append({
                            'type': 'image_statistics',
                            'image': base64.b64encode(img_bytes).decode()
                        })
                        print("✓ Generated enhanced image statistics plot")
                    except Exception as e:
                        print(f"Could not convert image_statistics to PNG with kaleido: {e}")
                        print("Trying matplotlib-based approach...")
                        # Fallback: Use matplotlib to render the plot data directly
                        try:
                            import matplotlib.pyplot as plt
                            import matplotlib
                            matplotlib.use('Agg')
                            
                            # Extract data from the Plotly figure and create matplotlib version
                            # For now, create a simpler version using the actual image data
                            image_cols = generator._find_image_columns(df)
                            if image_cols:
                                # Create a simplified statistics plot with matplotlib
                                fig_mpl, axes = plt.subplots(2, 2, figsize=(14, 10))
                                
                                # Get label column
                                label_col = None
                                for col in df.columns:
                                    if any(term in col.lower() for term in ['label', 'class', 'category', 'digit']):
                                        label_col = col
                                        break
                                
                                if label_col:
                                    # Plot 1: Class distribution
                                    class_counts = df[label_col].value_counts().sort_index()
                                    axes[0, 0].bar(range(len(class_counts)), class_counts.values, color='steelblue', alpha=0.7)
                                    axes[0, 0].set_xticks(range(len(class_counts)))
                                    axes[0, 0].set_xticklabels(class_counts.index, rotation=45, ha='right')
                                    axes[0, 0].set_xlabel('Class')
                                    axes[0, 0].set_ylabel('Count')
                                    axes[0, 0].set_title('Class Distribution')
                                    axes[0, 0].grid(True, alpha=0.3, axis='y')
                                
                                # Plot 2: Dataset info
                                axes[0, 1].axis('off')
                                info_text = f"""📸 Image Dataset Statistics

Total Images: {len(df)}
Classes: {df[label_col].nunique() if label_col else 'N/A'}
Columns: {len(df.columns)}

Image Columns: {', '.join(image_cols[:3]) if image_cols else 'None'}
"""
                                axes[0, 1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                                               bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.8))
                                
                                # Plot 3 & 4: Placeholder for future stats
                                axes[1, 0].axis('off')
                                axes[1, 0].text(0.5, 0.5, '📸 IMAGE DATA\nEnhanced statistics available\nvia chat: "Show image statistics"', 
                                               ha='center', va='center', fontsize=14,
                                               bbox=dict(boxstyle="round,pad=1", facecolor="lightgreen", alpha=0.8))
                                
                                axes[1, 1].axis('off')
                                axes[1, 1].text(0.5, 0.5, 'Use interactive chat to generate:\n• Color channel statistics\n• Quality metrics\n• Class separability', 
                                               ha='center', va='center', fontsize=12,
                                               bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.8))
                                
                                plt.suptitle('📸 Image Statistics Overview', fontsize=16, fontweight='bold', y=0.98)
                                plt.tight_layout()
                                
                                img_buffer = io.BytesIO()
                                plt.savefig(img_buffer, format='png', dpi=200, bbox_inches='tight')
                                plt.close()
                                img_buffer.seek(0)
                                base64_images.append({
                                    'type': 'image_statistics',
                                    'image': base64.b64encode(img_buffer.read()).decode()
                                })
                                print("✓ Generated matplotlib-based image statistics plot")
                            else:
                                print("No image columns found for statistics")
                        except Exception as e2:
                            print(f"Matplotlib fallback also failed: {e2}")
                            import traceback
                            traceback.print_exc()
                except Exception as e:
                    print(f"Image statistics plot failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 2. Embedding Visualization (using ResNet18 pre-trained model - small & efficient)
                try:
                    label_col = None
                    for col in df.columns:
                        if any(term in col.lower() for term in ['label', 'class', 'category', 'digit']):
                            label_col = col
                            break
                    
                    if label_col:
                        print("Generating embedding visualization using ResNet18 (ImageNet pre-trained)...")
                        plot_spec = {'plot_type': 'embedding_visualization', 'title': 'Image Embedding Visualization (ResNet18)'}
                        fig = generator._create_embedding_visualization(df, plot_spec, 'Embedding Space')
                        # Note: _create_embedding_visualization already calls _add_image_data_indicator internally
                        
                        try:
                            img_bytes = fig.to_image(format="png", width=1400, height=800, scale=2)
                            base64_images.append({
                                'type': 'embedding_visualization',
                                'image': base64.b64encode(img_bytes).decode()
                            })
                            print("✓ Generated embedding visualization (ResNet18 pre-trained)")
                        except Exception as e:
                            print(f"Could not convert embedding_visualization to PNG: {e}")
                            # Skip fallback for this one - it requires the model
                except Exception as e:
                    print(f"Embedding visualization plot failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 3. Class Separability (using ResNet18 embeddings - pre-trained model)
                try:
                    if label_col:
                        print("Generating class separability using ResNet18 embeddings...")
                        plot_spec = {'plot_type': 'class_separability', 'title': 'Class Separability Analysis (ResNet18)'}
                        fig = generator._create_class_separability(df, plot_spec, 'Class Separability')
                        # Note: _create_class_separability already calls _add_image_data_indicator internally at line 1302
                        
                        try:
                            img_bytes = fig.to_image(format="png", width=1400, height=1200, scale=2)
                            base64_images.append({
                                'type': 'class_separability',
                                'image': base64.b64encode(img_bytes).decode()
                            })
                            print("✓ Generated enhanced class separability plot (ResNet18 pre-trained)")
                        except Exception as e:
                            print(f"Could not convert class_separability to PNG with kaleido: {e}")
                    else:
                        print("No label column found for class separability")
                except Exception as e:
                    print(f"Class separability plot failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 3b. Class Similarity Matrix (NEW - using ResNet18)
                try:
                    if label_col:
                        print("Generating inter-class similarity matrix using ResNet18...")
                        plot_spec = {'plot_type': 'class_similarity_matrix', 'title': 'Inter-Class Similarity (ResNet18)'}
                        fig = generator._create_class_similarity_matrix(df, plot_spec, 'Class Similarity')
                        
                        try:
                            img_bytes = fig.to_image(format="png", width=1400, height=800, scale=2)
                            base64_images.append({
                                'type': 'class_similarity_matrix',
                                'image': base64.b64encode(img_bytes).decode()
                            })
                            print("✓ Generated class similarity matrix (ResNet18 pre-trained)")
                        except Exception as e:
                            print(f"Could not convert class_similarity_matrix to PNG: {e}")
                    else:
                        print("No label column found for class similarity matrix")
                except Exception as e:
                    print(f"Class similarity matrix plot failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 4. Image Grid (Sample Images)
                try:
                    image_cols = generator._find_image_columns(df)
                    if image_cols and label_col:
                        plot_spec = {'plot_type': 'image_grid', 'title': 'Sample Images by Class'}
                        # Use the generator's image grid method
                        html_content = generator._create_image_grid(df, plot_spec, 'Sample Images')
                        # For automatic display, create a simple matplotlib version
                        import matplotlib.pyplot as plt
                        import matplotlib
                        matplotlib.use('Agg')
                        from PIL import Image as PILImage
                        
                        image_col = image_cols[0]
                        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                        axes = axes.ravel()
                        
                        sample_count = 0
                        for label in sorted(df[label_col].unique())[:10]:
                            if sample_count >= 10:
                                break
                            sample = df[df[label_col] == label].head(1)
                            if len(sample) > 0:
                                try:
                                    img_data = str(sample[image_col].iloc[0])
                                    if img_data.startswith('data:image') or img_data.startswith('iVBORw0KGgo'):
                                        base64_data = img_data.split(',')[1] if ',' in img_data else img_data
                                        img_bytes = base64.b64decode(base64_data)
                                        img = PILImage.open(io.BytesIO(img_bytes))
                                        axes[sample_count].imshow(img, cmap='gray' if img.mode == 'L' else None)
                                        axes[sample_count].set_title(f'Class: {label}', fontsize=10)
                                        axes[sample_count].axis('off')
                                        sample_count += 1
                                except Exception as e:
                                    print(f"Could not display image for class {label}: {e}")
                        
                        for i in range(sample_count, 10):
                            axes[i].axis('off')
                        
                        plt.suptitle('📸 Sample Images by Class', fontsize=14, fontweight='bold', y=1.02)
                        plt.tight_layout()
                        
                        img_buffer = io.BytesIO()
                        plt.savefig(img_buffer, format='png', dpi=200, bbox_inches='tight')
                        plt.close()
                        img_buffer.seek(0)
                        base64_images.append({
                            'type': 'image_grid',
                            'image': base64.b64encode(img_buffer.read()).decode()
                        })
                        print("✓ Generated image grid plot")
                except Exception as e:
                    print(f"Image grid plot failed: {e}")
                
                if len(base64_images) == 0:
                    print("WARNING: No image visualizations generated, creating fallback...")
                    # Create a simple fallback plot
                    try:
                        plt.figure(figsize=(10, 6))
                        plt.text(0.5, 0.5, f'Image Data Loaded\nTotal Images: {len(df)}\nColumns: {", ".join(df.columns[:10])}', 
                                ha='center', va='center', fontsize=12, transform=plt.gca().transAxes,
                                bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.8))
                        plt.title('📸 Image Data Overview', fontsize=14, fontweight='bold')
                        plt.axis('off')
                        fallback_path = os.path.join(user_results_dir, f"image_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                        plt.savefig(fallback_path, dpi=200, bbox_inches='tight')
                        plt.close()
                        with open(fallback_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                        base64_images.append({
                            'type': 'image_overview',
                            'image': img_data
                        })
                        print("✓ Created fallback image plot")
                    except Exception as e:
                        print(f"ERROR: Failed to create fallback: {e}")
                        return jsonify({
                            'user_id': user_id,
                            'data_type': data_type,
                            'images': [],
                            'error': f'Failed to generate image visualizations: {str(e)}'
                        }), 500
                
                print(f"DEBUG: Generated {len(base64_images)} image visualizations")
                print(f"DEBUG: Image types: {[img.get('type', 'unknown') for img in base64_images]}")
                return jsonify({
                    'user_id': user_id,
                    'data_type': data_type,
                    'images': base64_images,
                    'message': f'Image data statistics generated successfully - {len(base64_images)} visualizations'
                })
            except Exception as e:
                print(f"ERROR in image data statistics: {e}")
                import traceback
                traceback.print_exc()
                # Return error instead of falling through
                return jsonify({
                    'user_id': user_id,
                    'data_type': data_type,
                    'images': [],
                    'error': f'Error generating image statistics: {str(e)}'
                }), 500
        
        # Create user-specific results directory
        import os
        from datetime import datetime
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        import glob
        import base64
        
        user_results_dir = f"/app/shared_data/results/{user_id}"
        os.makedirs(user_results_dir, exist_ok=True)
        
        # Clear old data statistics files for this user
        old_files = glob.glob(os.path.join(user_results_dir, "overview_*.png"))
        old_files.extend(glob.glob(os.path.join(user_results_dir, "sentiment_distribution_*.png")))
        old_files.extend(glob.glob(os.path.join(user_results_dir, "per_asset_sentiment_*.png")))
        old_files.extend(glob.glob(os.path.join(user_results_dir, "keyword_insights_*.png")))
        old_files.extend(glob.glob(os.path.join(user_results_dir, "word_sentiment_associations_*.png")))
        old_files.extend(glob.glob(os.path.join(user_results_dir, "asset_distribution_*.png")))
        old_files.extend(glob.glob(os.path.join(user_results_dir, "data_quality_*.png")))
        
        for old_file in old_files:
            try:
                os.remove(old_file)
                print(f"Removed old data statistics file: {old_file}")
            except Exception as e:
                print(f"Could not remove old file {old_file}: {e}")
        
        images = []
        
        # Initialize plot data variables
        word_sentiment_data = None
        keyword_data = None
        
        # Find title column
        title_col = None
        possible_title_cols = ['title', 'headline', 'text', 'content', 'article']
        for col in df.columns:
            if any(title_word in col.lower() for title_word in possible_title_cols):
                title_col = col
                break
        
        if not title_col:
            # Use first text column
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            title_col = text_cols[0] if text_cols else None
        
        if not title_col:
            return jsonify({'error': 'No text column found in data'}), 400
        
        # Find sentiment column
        sentiment_col = None
        for col in df.columns:
            if 'sentiment' in col.lower():
                sentiment_col = col
                break
        
        # Find asset column
        asset_col = None
        for col in df.columns:
            if 'asset' in col.lower() or 'ticker' in col.lower() or 'symbol' in col.lower():
                asset_col = col
                break
        
        # Collect overview statistics (no plot, just data for frontend)
        overview_stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'data_type': data_type.upper(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'column_names': list(df.columns[:20])  # First 20 columns
        }
        
        # Add data-specific insights if available
        if title_col:
            overview_stats['avg_text_length'] = round(df[title_col].astype(str).str.len().mean(), 1)
        
        if sentiment_col:
            overview_stats['sentiment_avg'] = round(df[sentiment_col].mean(), 3)
            overview_stats['sentiment_min'] = round(df[sentiment_col].min(), 3)
            overview_stats['sentiment_max'] = round(df[sentiment_col].max(), 3)
        
        if asset_col:
            overview_stats['unique_assets'] = df[asset_col].nunique()
        
        print(f"✓ Collected overview statistics: {overview_stats}")
        
        # Only generate additional plots if explicitly requested
        # If no plot_types specified, only return the basic overview
        if generate_all or len(plot_types) > 0:
            # Generate requested plots or all plots if generate_all
            requested_plots = plot_types if plot_types else ['sentiment_distribution', 'keyword_insights', 'asset_distribution', 'data_quality']
        
            # 2. Sentiment Distribution (if available and requested)
            if (generate_all or 'sentiment_distribution' in plot_types) and sentiment_col:
                try:
                    plt.figure(figsize=(12, 6))
                    plt.hist(df[sentiment_col], bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
                    plt.title('Sentiment Distribution (Titles)', fontsize=14, fontweight='bold')
                    plt.xlabel('Sentiment Score', fontsize=12)
                    plt.ylabel('Count', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    sentiment_dist_path = os.path.join(user_results_dir, f"sentiment_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    plt.savefig(sentiment_dist_path, dpi=200, bbox_inches='tight')
                    plt.close()
                    images.append({'type': 'sentiment_distribution', 'file': sentiment_dist_path})
                    print(f"✓ Generated sentiment distribution plot: {sentiment_dist_path}")
                except Exception as e:
                    print(f"ERROR: Sentiment distribution failed: {e}")
                    import traceback
                    traceback.print_exc()
        
            # 3. Per-Asset Sentiment (if both sentiment and asset columns exist and requested)
            if (generate_all or 'per_asset_sentiment' in plot_types) and sentiment_col and asset_col:
                try:
                    # Limit to top 25 assets by article count for readability
                    asset_counts = df[asset_col].value_counts()
                    top_assets = asset_counts.head(25).index
                    filtered_df = df[df[asset_col].isin(top_assets)]
                    
                    plt.figure(figsize=(16, 10))
                    filtered_df.boxplot(column=sentiment_col, by=asset_col, rot=45)
                    plt.title('Per-Asset Sentiment (Top 25 Assets by Article Count)', fontsize=14, fontweight='bold')
                    plt.suptitle('')
                    plt.xlabel('Asset', fontsize=12)
                    plt.ylabel('Sentiment Score', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    per_asset_path = os.path.join(user_results_dir, f"per_asset_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    plt.savefig(per_asset_path, dpi=200, bbox_inches='tight')
                    plt.close()
                    images.append({'type': 'per_asset_sentiment', 'file': per_asset_path})
                except Exception as e:
                    print(f"Per-asset sentiment failed: {e}")
        
            # 4. Keyword Insights (Top words in titles) - if requested
            if generate_all or 'keyword_insights' in plot_types:
                try:
                    from sklearn.feature_extraction.text import CountVectorizer
                    
                    cv = CountVectorizer(stop_words='english', max_features=200)
                    X = cv.fit_transform(df[title_col])
                    freqs = zip(cv.get_feature_names_out(), X.sum(axis=0).A1)
                    top = sorted(freqs, key=lambda x: x[1], reverse=True)[:15]
                    
                    if top:
                        words, counts = zip(*top)
                        
                        plt.figure(figsize=(12, 8))
                        colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
                        bars = plt.barh(range(len(words)), counts, color=colors)
                        plt.yticks(range(len(words)), words)
                        plt.xlabel('Frequency in Article Titles', fontsize=12)
                        plt.title('Top 15 Keywords in Titles', fontsize=14, fontweight='bold')
                        plt.grid(True, alpha=0.3)
                        plt.gca().invert_yaxis()
                        
                        # Add value labels on bars
                        for i, (bar, count) in enumerate(zip(bars, counts)):
                            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                                   str(count), ha='left', va='center', fontsize=10)
                        
                        plt.tight_layout()
                        keyword_path = os.path.join(user_results_dir, f"keyword_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                        plt.savefig(keyword_path, dpi=200, bbox_inches='tight')
                        plt.close()
                        images.append({'type': 'keyword_insights', 'file': keyword_path})
                        
                        # Store the actual keyword data for AI assistant access
                        keyword_data = {
                            'top_keywords': [(word, int(count)) for word, count in top]
                        }
                except Exception as e:
                    print(f"Keyword insights failed: {e}")
        
            # 5. Word-Sentiment Associations (if sentiment column exists and requested)
            if (generate_all or 'word_sentiment_associations' in plot_types) and sentiment_col:
                try:
                    cv = CountVectorizer(stop_words='english')
                    X = cv.fit_transform(df[title_col])
                    words = cv.get_feature_names_out()
                    sentiments = df[sentiment_col].fillna(0).to_numpy()
                    total_sent = X.T.dot(sentiments)
                    
                    if hasattr(total_sent, 'A1'):
                        sent_arr = total_sent.A1
                    else:
                        sent_arr = np.asarray(total_sent).reshape(-1)
                    
                    pairs = list(zip(words, sent_arr))
                    pos = sorted(pairs, key=lambda x: x[1], reverse=True)[:10]
                    neg = sorted(pairs, key=lambda x: x[1])[:10]
                    
                    # Create subplot for positive and negative words
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Positive words
                    pos_words, pos_scores = zip(*pos)
                    bars1 = ax1.barh(range(len(pos_words)), pos_scores, color='green', alpha=0.7)
                    ax1.set_yticks(range(len(pos_words)))
                    ax1.set_yticklabels(pos_words)
                    ax1.set_xlabel('Sentiment Score', fontsize=12)
                    ax1.set_title('Top 10 Words Driving Positive Sentiment', fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    ax1.invert_yaxis()
                    
                    # Add value labels
                    for i, (bar, score) in enumerate(zip(bars1, pos_scores)):
                        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                                f'{score:.2f}', ha='left', va='center', fontsize=10)
                    
                    # Negative words
                    neg_words, neg_scores = zip(*neg)
                    bars2 = ax2.barh(range(len(neg_words)), neg_scores, color='red', alpha=0.7)
                    ax2.set_yticks(range(len(neg_words)))
                    ax2.set_yticklabels(neg_words)
                    ax2.set_xlabel('Sentiment Score', fontsize=12)
                    ax2.set_title('Top 10 Words Driving Negative Sentiment', fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    ax2.invert_yaxis()
                    
                    # Add value labels
                    for i, (bar, score) in enumerate(zip(bars2, neg_scores)):
                        ax2.text(bar.get_width() - 0.01, bar.get_y() + bar.get_height()/2, 
                                f'{score:.2f}', ha='right', va='center', fontsize=10)
                    
                    plt.tight_layout()
                    word_sentiment_path = os.path.join(user_results_dir, f"word_sentiment_associations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    plt.savefig(word_sentiment_path, dpi=200, bbox_inches='tight')
                    plt.close()
                    images.append({'type': 'word_sentiment_associations', 'file': word_sentiment_path})
                    
                    # Store the actual word sentiment data for AI assistant access
                    word_sentiment_data = {
                        'positive_words': [(word, float(score)) for word, score in pos],
                        'negative_words': [(word, float(score)) for word, score in neg]
                    }
                    import sys
                    sys.stdout.write(f"DEBUG: word_sentiment_data created with {len(word_sentiment_data['positive_words'])} positive and {len(word_sentiment_data['negative_words'])} negative words\n")
                    sys.stdout.flush()
                except Exception as e:
                    import sys
                    sys.stdout.write(f"Word sentiment associations failed: {e}\n")
                    sys.stdout.flush()
                    import traceback
                    traceback.print_exc()
        
            # 6. Asset Distribution (if asset column exists and requested)
            if (generate_all or 'asset_distribution' in plot_types) and asset_col:
                try:
                    asset_counts = df[asset_col].value_counts().head(20)
                    plt.figure(figsize=(14, 10))
                    colors = plt.cm.Set3(np.linspace(0, 1, len(asset_counts)))
                    bars = plt.barh(range(len(asset_counts)), asset_counts.values, color=colors)
                    plt.yticks(range(len(asset_counts)), asset_counts.index)
                    plt.xlabel('Number of Articles', fontsize=12)
                    plt.title('Top 20 Assets by Article Count', fontsize=14, fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    plt.gca().invert_yaxis()
                    
                    # Add value labels
                    for i, (bar, count) in enumerate(zip(bars, asset_counts.values)):
                        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                               str(count), ha='left', va='center', fontsize=10)
                    
                    plt.tight_layout()
                    asset_dist_path = os.path.join(user_results_dir, f"asset_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    plt.savefig(asset_dist_path, dpi=200, bbox_inches='tight')
                    plt.close()
                    images.append({'type': 'asset_distribution', 'file': asset_dist_path})
                except Exception as e:
                    print(f"Asset distribution failed: {e}")
        

            # 7. Data Quality - Text Length Distribution (if requested)
            if generate_all or 'data_quality' in plot_types:
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # Calculate actual text lengths from the data
                    if title_col:
                        text_lengths = df[title_col].astype(str).str.len()
                    else:
                        # Fallback to simulated data if no title column
                        text_lengths = pd.Series(np.random.lognormal(mean=4, sigma=0.5, size=len(df)))
                    
                    # Create histogram
                    plt.hist(text_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.xlabel('Text Length (characters)', fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                    plt.title('Text Length Distribution', fontsize=14, fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    
                    # Add statistics
                    mean_len = text_lengths.mean()
                    std_len = text_lengths.std()
                    median_len = text_lengths.median()
                    
                    plt.axvline(mean_len, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_len:.1f}')
                    plt.axvline(median_len, color='green', linestyle='--', linewidth=2, label=f'Median: {median_len:.1f}')
                    plt.axvline(mean_len + std_len, color='orange', linestyle='--', linewidth=2, label=f'+1σ: {mean_len + std_len:.1f}')
                    plt.axvline(mean_len - std_len, color='orange', linestyle='--', linewidth=2, label=f'-1σ: {mean_len - std_len:.1f}')
                    plt.legend()
                    
                    # Add summary statistics text box
                    stats_text = f"""Statistics:
• Count: {len(text_lengths):,}
• Mean: {mean_len:.1f}
• Median: {median_len:.1f}
• Std Dev: {std_len:.1f}
• Min: {text_lengths.min():.1f}
• Max: {text_lengths.max():.1f}"""
                    
                    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
                    
                    plt.tight_layout()
                    data_quality_path = os.path.join(user_results_dir, f"data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    plt.savefig(data_quality_path, dpi=200, bbox_inches='tight')
                    plt.close()
                    images.append({'type': 'data_quality', 'file': data_quality_path})
                except Exception as e:
                    print(f"Data quality visualization failed: {e}")
        
        # Convert file paths to base64 for frontend display and AI outputs service
        print(f"DEBUG: Converting {len(images)} images to base64")
        base64_images = []
        for img in images:
            if 'file' in img:
                try:
                    if os.path.exists(img['file']):
                        with open(img['file'], 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode()
                        base64_images.append({
                            'type': img['type'],
                            'image': img_data
                        })
                        print(f"✓ Converted {img['type']} to base64")
                    else:
                        print(f"ERROR: File not found: {img['file']}")
                except Exception as e:
                    print(f"ERROR: Error converting {img['file']} to base64: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                base64_images.append(img)
        
        print(f"DEBUG: Total base64 images: {len(base64_images)}")
        
        # Ensure we have at least one plot
        if len(base64_images) == 0:
            print("WARNING: No plots generated! Creating fallback plot...")
            try:
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f'No visualizations could be generated.\nData shape: {df.shape}\nColumns: {", ".join(df.columns[:10])}', 
                        ha='center', va='center', fontsize=12, transform=plt.gca().transAxes,
                        bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.8))
                plt.title('Data Statistics - No Plots Available', fontsize=14, fontweight='bold')
                plt.axis('off')
                fallback_path = os.path.join(user_results_dir, f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(fallback_path, dpi=200, bbox_inches='tight')
                plt.close()
                with open(fallback_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                base64_images.append({
                    'type': 'overview',
                    'image': img_data
                })
                print("✓ Created fallback plot")
            except Exception as e:
                print(f"ERROR: Failed to create fallback plot: {e}")
        
        # Store data statistics in AI outputs service for AI assistant access
        try:
            import requests
            from datetime import datetime
            
            # Create visualization names for AI assistant access
            viz_names = []
            for img in base64_images:
                if 'type' in img:
                    viz_names.append(img['type'])
            
            # Collect all the actual data for AI assistant access
            plot_data = {}
            if word_sentiment_data is not None:
                plot_data['word_sentiment'] = word_sentiment_data
                import sys
                sys.stdout.write(f"DEBUG: Added word_sentiment to plot_data: {len(word_sentiment_data.get('positive_words', []))} positive, {len(word_sentiment_data.get('negative_words', []))} negative words\n")
                sys.stdout.flush()
            if keyword_data is not None:
                plot_data['keywords'] = keyword_data
                import sys
                sys.stdout.write("DEBUG: Added keywords to plot_data\n")
                sys.stdout.flush()
            import sys
            sys.stdout.write(f"DEBUG: Final plot_data keys: {list(plot_data.keys())}\n")
            sys.stdout.flush()

            plot_summaries = []
            if 'word_sentiment' in plot_data:
                plot_summaries.append({
                    'title': 'Word Sentiment Associations',
                    'plot_type': 'word_sentiment_association',
                    'description': 'Top words driving positive and negative sentiment.',
                    'data': plot_data['word_sentiment'],
                    'summary_text': 'Top positive and negative sentiment-associated words.'
                })
            if 'keywords' in plot_data:
                plot_summaries.append({
                    'title': 'Top Keywords',
                    'plot_type': 'keyword_frequency',
                    'description': 'Most frequent words in titles.',
                    'data': plot_data['keywords'],
                    'summary_text': 'Most frequent keywords in the dataset.'
                })
            
            data_statistics_payload = {
                'user_id': user_id,
                'data_statistics': {
                    'data_type': data_type,
                    'visualizations': viz_names,
                    'analysis_type': 'data_statistics',
                    'timestamp': datetime.now().isoformat(),
                    'plot_data': plot_data,  # Include the actual plot data
                    'plot_summaries': plot_summaries
                },
                'images': base64_images,  # Include the base64 images
                'plot_summaries': plot_summaries
            }
            ai_outputs_url = f'{AI_OUTPUTS_SERVICE_URL}/store-results'
            response = requests.post(ai_outputs_url, json=data_statistics_payload, timeout=10)
            if response.status_code == 200:
                logger.info("Successfully stored data statistics in AI outputs service for user %s", user_id)
            else:
                logger.warning("Failed to store data statistics in AI outputs service: %s %s", response.status_code, response.text)
        except Exception as e:
            logger.warning("Could not store data statistics in AI outputs service: %s", e)
        
        print(f"=== DATA STATISTICS SUCCESS ===")
        print(f"Generated {len(base64_images)} visualizations")
        print(f"Types: {[img.get('type', 'unknown') for img in base64_images]}")
        
        return jsonify({
            'user_id': user_id, 
            'data_type': data_type, 
            'images': base64_images, 
            'message': f'Data statistics generated successfully - {len(base64_images)} visualizations',
            'overview_stats': overview_stats,  # Include overview statistics for frontend
            'data_shape': f"{overview_stats['total_rows']:,} × {overview_stats['total_columns']}",
            'columns_count': overview_stats['total_columns']
        })
    except Exception as e:
        print(f"=== ERROR in data_statistics ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generating data statistics: {str(e)}'}), 500

# --- Enhanced XAI Methods ---
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import io, base64

def generate_enhanced_attention_analysis(example_text, user_id):
    """
    Generate enhanced attention analysis with dual-panel layout, token importance,
    and detailed insights for AI assistant access.
    """
    try:
        print("Starting enhanced attention analysis...", flush=True)
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('/app/shared_data/models/ProsusAI/finbert')
        model = AutoModelForSequenceClassification.from_pretrained(
            '/app/shared_data/models/ProsusAI/finbert', output_attentions=True
        )
        
        # Tokenize input
        inputs = tokenizer(example_text, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = model(**inputs)
            attention = outputs.attentions[-1][0]  # Last layer attention
        
        # Calculate average attention across all heads
        avg_attention = attention.mean(dim=0).detach().numpy()
        
        # Calculate token importance scores (sum of attention received by each token)
        token_importance = avg_attention.sum(axis=0)
        
        # Get top 10 most important tokens
        top_tokens_idx = np.argsort(token_importance)[-10:][::-1]
        top_tokens = [(tokens[i], token_importance[i]) for i in top_tokens_idx]
        
        # Calculate attention concentration metrics
        attention_variance = np.var(avg_attention)
        attention_entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-10))
        max_attention_score = np.max(avg_attention)
        
        # Create dual-panel visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Panel 1: Enhanced Attention Heatmap
        ax1 = plt.subplot(2, 2, (1, 2))
        im = ax1.imshow(avg_attention, cmap='viridis', aspect='auto')
        
        # Color-coded annotations for maximum attention scores
        max_attention_pos = np.unravel_index(np.argmax(avg_attention), avg_attention.shape)
        ax1.plot(max_attention_pos[1], max_attention_pos[0], 'r*', markersize=15, label=f'Max Attention: {max_attention_score:.3f}')
        
        # Set labels and title
        ax1.set_xticks(range(len(tokens)))
        ax1.set_yticks(range(len(tokens)))
        ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels(tokens, fontsize=8)
        ax1.set_title(f'Enhanced Attention Heatmap\nText: "{example_text[:80]}{"..." if len(example_text) > 80 else ""}"', 
                      fontsize=12, fontweight='bold')
        ax1.legend()
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Attention Score', fontsize=10)
        
        # Panel 2: Token Importance Bar Chart
        ax2 = plt.subplot(2, 2, 3)
        top_tokens_names = [token for token, _ in top_tokens]
        top_tokens_scores = [score for _, score in top_tokens]
        
        bars = ax2.barh(range(len(top_tokens_names)), top_tokens_scores, 
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_tokens_names))))
        
        # Add value annotations on bars
        for i, (bar, score) in enumerate(zip(bars, top_tokens_scores)):
            ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', ha='left', fontsize=9)
        
        ax2.set_yticks(range(len(top_tokens_names)))
        ax2.set_yticklabels(top_tokens_names, fontsize=9)
        ax2.set_xlabel('Importance Score', fontsize=10)
        ax2.set_title('Top 10 Most Important Tokens', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Attention Distribution Analysis
        ax3 = plt.subplot(2, 2, 4)
        
        # Flatten attention matrix for distribution analysis
        attention_flat = avg_attention.flatten()
        
        # Create histogram of attention scores
        ax3.hist(attention_flat, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(max_attention_score, color='red', linestyle='--', 
                    label=f'Max: {max_attention_score:.3f}')
        ax3.axvline(np.mean(attention_flat), color='green', linestyle='--', 
                    label=f'Mean: {np.mean(attention_flat):.3f}')
        
        ax3.set_xlabel('Attention Score', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title('Attention Score Distribution', fontsize=11, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save enhanced attention plot in user-specific directory
        user_results_dir = f"/app/shared_data/results/{user_id}"
        os.makedirs(user_results_dir, exist_ok=True)
        attention_path = os.path.join(user_results_dir, f"enhanced_attention_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(attention_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return file path instead of base64
        attention_img = attention_path
        
        # Generate detailed insights for AI assistant
        attention_insights = {
            'top_tokens': [(str(token), float(score)) for token, score in top_tokens],
            'max_attention_score': float(max_attention_score),
            'attention_variance': float(attention_variance),
            'attention_entropy': float(attention_entropy),
            'attention_concentration': 'concentrated' if attention_variance > 0.1 else 'distributed',
            'sentiment_correlation': 'positive' if any('good' in token.lower() or 'positive' in token.lower() for token, _ in top_tokens[:3]) else 
                                   'negative' if any('bad' in token.lower() or 'negative' in token.lower() for token, _ in top_tokens[:3]) else 'neutral'
        }
        
        # Store attention insights in AI outputs service
        try:
            import requests
            import json
            
            # Ensure all values are JSON serializable
            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, tuple):
                    return [make_json_serializable(item) for item in obj]
                elif hasattr(obj, 'dtype'):  # numpy types
                    return float(obj)
                else:
                    return obj
            
            serializable_insights = make_json_serializable(attention_insights)
            
            attention_data = {
                'user_id': user_id,
                'attention_analysis': {
                    'example_text': example_text,
                    'insights': serializable_insights,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            ai_outputs_url = f'{AI_OUTPUTS_SERVICE_URL}/store-attention-insights'
            response = requests.post(ai_outputs_url, json=attention_data, timeout=10)
            if response.status_code == 200:
                logger.info("Attention insights stored in AI outputs service successfully")
            else:
                logger.warning("Failed to store attention insights: %s", response.status_code)
        except Exception as e:
            logger.warning("Could not store attention insights: %s", e)
        
        print("Enhanced attention analysis completed successfully", flush=True)
        return attention_img, attention_insights
        
    except Exception as e:
        import traceback
        print(f"ERROR in enhanced attention analysis: {e}", flush=True)
        traceback.print_exc()
        return None, None

def explain_model_type(model):
    """Return a string explaining the model type and its capabilities."""
    model_type = type(model).__name__
    if 'RandomForest' in model_type:
        return f"{model_type}: An ensemble of decision trees, robust to overfitting, provides feature importances."
    elif 'LinearRegression' in model_type or 'Ridge' in model_type:
        return f"{model_type}: A linear model, interpretable coefficients, assumes linear relationships."
    elif 'XGB' in model_type or 'LGBM' in model_type:
        return f"{model_type}: Gradient boosting model, powerful for tabular data, provides SHAP explanations."
    else:
        return f"{model_type}: Model type not specifically documented."

def generate_shap_summary_plot(model, X, feature_names):
    """Generate a SHAP summary plot for global feature importance."""
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=200, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        return {'type': 'shap_summary', 'image': base64.b64encode(img_buffer.getvalue()).decode()}
    except Exception as e:
        print(f"SHAP summary plot failed: {e}")
        return None

def generate_lime_explanation(model, X, feature_names, sample_idx=0):
    """Generate a LIME explanation for a specific sample."""
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=feature_names, class_names=['Prediction'], mode='regression')
        exp = explainer.explain_instance(X.values[sample_idx], model.predict, num_features=min(10, len(feature_names)))
        fig = exp.as_pyplot_figure()
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=200, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        return {'type': 'lime_explanation', 'image': base64.b64encode(img_buffer.getvalue()).decode(), 'sample_idx': sample_idx}
    except Exception as e:
        print(f"LIME explanation failed: {e}")
        return None

def generate_model_documentation(model, X, feature_names):
    """Return a string with model documentation and training data summary."""
    doc = explain_model_type(model)
    doc += f"\nTraining data shape: {X.shape}\nFeatures: {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}"
    if hasattr(model, 'score'):
        try:
            score = model.score(X, model.predict(X))
            doc += f"\nModel score (on training data): {score:.3f}"
        except Exception:
            pass
    return doc

def generate_advanced_xai(model, X, feature_names, user_id):
    """Generate advanced XAI visualizations and explanations."""
    results = []
    # Model documentation
    doc = generate_model_documentation(model, X, feature_names)
    results.append({'type': 'model_documentation', 'text': doc})
    # SHAP summary
    shap_img = generate_shap_summary_plot(model, X, feature_names)
    if shap_img:
        results.append(shap_img)
    # LIME for first sample
    lime_img = generate_lime_explanation(model, X, feature_names, sample_idx=0)
    if lime_img:
        results.append(lime_img)
    # LIME for a random sample (if more than 1 row)
    if X.shape[0] > 1:
        import numpy as np
        idx = np.random.randint(1, X.shape[0])
        lime_img2 = generate_lime_explanation(model, X, feature_names, sample_idx=idx)
        if lime_img2:
            results.append(lime_img2)
    return results

def generate_raw_close_analysis(model, df, user_id):
    """Generate XAI analysis for raw close values without preprocessing"""
    results = []
    
    try:
        # Get numeric columns (close values)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            return results
        
        # Use first 5 stocks for analysis
        analysis_cols = numeric_cols[:5]
        X_raw = df[analysis_cols].fillna(method='ffill').fillna(0)
        
        # 1. Lag-based Feature Importance
        lag_importance_viz = generate_lag_importance_analysis(model, X_raw, analysis_cols, user_id)
        if lag_importance_viz:
            results.append({
                'type': 'image',
                'image': lag_importance_viz
            })
        
        # 2. Counterfactual Examples
        counterfactual_viz = generate_counterfactual_examples(model, X_raw, analysis_cols, user_id)
        if counterfactual_viz:
            results.append({
                'type': 'image',
                'image': counterfactual_viz
            })
        
        # 3. Individual Prediction Explanations
        individual_viz = generate_individual_predictions(model, X_raw, analysis_cols, user_id)
        if individual_viz:
            results.append({
                'type': 'image',
                'image': individual_viz
            })
        
        return results
    except Exception as e:
        print(f"Error in generate_raw_close_analysis: {e}")
        import traceback
        traceback.print_exc()
        return []

def generate_lag_importance_analysis(model, X, feature_names, user_id):
    """Generate lag-based feature importance analysis"""
    try:
        # Create lag features for analysis
        X_with_lags = X.copy()
        for col in feature_names[:3]:  # Use first 3 stocks
            for lag in [1, 2, 3, 5, 7]:
                X_with_lags[f'{col}_lag{lag}'] = X_with_lags[col].shift(lag)
        
        X_with_lags = X_with_lags.fillna(0)
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            # Use permutation importance
            from sklearn.inspection import permutation_importance
            result = permutation_importance(model, X_with_lags.iloc[-100:], 
                                         np.random.randn(100), n_repeats=5, random_state=42)
            importances = result.importances_mean
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Group features by stock and lag
        feature_groups = {}
        for i, feature in enumerate(X_with_lags.columns):
            if '_lag' in feature:
                stock = feature.split('_lag')[0]
                lag = int(feature.split('_lag')[1])
                if stock not in feature_groups:
                    feature_groups[stock] = {}
                feature_groups[stock][lag] = importances[i] if i < len(importances) else 0
            else:
                stock = feature
                if stock not in feature_groups:
                    feature_groups[stock] = {}
                feature_groups[stock][0] = importances[i] if i < len(importances) else 0
        
        # Plot lag importance for each stock
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (stock, lags) in enumerate(feature_groups.items()):
            if idx >= 4:
                break
            
            lag_values = list(lags.keys())
            importance_values = list(lags.values())
            
            axes[idx].bar(lag_values, importance_values, alpha=0.7, color='skyblue')
            axes[idx].set_title(f'{stock} - Lag Importance')
            axes[idx].set_xlabel('Lag (days)')
            axes[idx].set_ylabel('Importance')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return {
            'type': 'image/png',
            'data': img_base64,
            'title': 'Lag-Based Feature Importance Analysis',
            'description': 'Shows how different lag periods (previous days) affect model predictions for each stock.'
        }
        
    except Exception as e:
        print(f"Error in generate_lag_importance_analysis: {e}")
        return None

def generate_counterfactual_examples(model, X, feature_names, user_id):
    """Generate counterfactual examples - what-if scenarios"""
    try:
        # Select a sample for counterfactual analysis
        sample_idx = min(100, len(X) - 1)
        original_sample = X.iloc[sample_idx:sample_idx+1]
        
        # Get original prediction
        original_pred = model.predict(original_sample)[0]
        
        # Create counterfactual scenarios
        scenarios = []
        for col in feature_names[:3]:  # Use first 3 stocks
            # Scenario 1: 10% increase
            scenario1 = original_sample.copy()
            scenario1[col] = scenario1[col] * 1.1
            pred1 = model.predict(scenario1)[0]
            scenarios.append({
                'stock': col,
                'change': '+10%',
                'original': original_sample[col].iloc[0],
                'new': scenario1[col].iloc[0],
                'prediction_change': pred1 - original_pred
            })
            
            # Scenario 2: 10% decrease
            scenario2 = original_sample.copy()
            scenario2[col] = scenario2[col] * 0.9
            pred2 = model.predict(scenario2)[0]
            scenarios.append({
                'stock': col,
                'change': '-10%',
                'original': original_sample[col].iloc[0],
                'new': scenario2[col].iloc[0],
                'prediction_change': pred2 - original_pred
            })
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Plot counterfactual scenarios
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Price changes
        stocks = [s['stock'] for s in scenarios[::2]]
        changes = [s['change'] for s in scenarios[::2]]
        price_changes = [(s['new'] - s['original']) / s['original'] * 100 for s in scenarios[::2]]
        
        bars1 = ax1.bar(range(len(stocks)), price_changes, color=['green' if c == '+10%' else 'red' for c in changes])
        ax1.set_title('Counterfactual Price Changes')
        ax1.set_xlabel('Stock')
        ax1.set_ylabel('Price Change (%)')
        ax1.set_xticks(range(len(stocks)))
        ax1.set_xticklabels(stocks, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Prediction changes
        pred_changes = [s['prediction_change'] for s in scenarios[::2]]
        bars2 = ax2.bar(range(len(stocks)), pred_changes, color=['green' if c == '+10%' else 'red' for c in changes])
        ax2.set_title('Prediction Changes')
        ax2.set_xlabel('Stock')
        ax2.set_ylabel('Prediction Change')
        ax2.set_xticks(range(len(stocks)))
        ax2.set_xticklabels(stocks, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return {
            'type': 'image/png',
            'data': img_base64,
            'title': 'Counterfactual Analysis',
            'description': f'Shows how predictions change when stock prices change by ±10%. Original prediction: {original_pred:.2f}'
        }
        
    except Exception as e:
        print(f"Error in generate_counterfactual_examples: {e}")
        return None

def generate_individual_predictions(model, X, feature_names, user_id):
    """Generate individual prediction explanations using LIME"""
    try:
        # Select a few samples for individual analysis
        sample_indices = [0, 50, 100, 150]
        sample_indices = [i for i in sample_indices if i < len(X)]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, sample_idx in enumerate(sample_indices):
            if idx >= 4:
                break
                
            sample = X.iloc[sample_idx:sample_idx+1]
            
            # Get feature contributions using LIME
            try:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X.values, 
                    feature_names=feature_names,
                    class_names=['prediction'],
                    mode='regression'
                )
                
                exp = explainer.explain_instance(
                    sample.values[0], 
                    model.predict, 
                    num_features=min(5, len(feature_names))
                )
                
                # Extract feature contributions
                features = [x[0] for x in exp.as_list()]
                contributions = [x[1] for x in exp.as_list()]
                
                # Plot
                colors = ['green' if c > 0 else 'red' for c in contributions]
                bars = axes[idx].barh(range(len(features)), contributions, color=colors, alpha=0.7)
                axes[idx].set_yticks(range(len(features)))
                axes[idx].set_yticklabels(features)
                axes[idx].set_title(f'Sample {sample_idx} - Individual Prediction')
                axes[idx].set_xlabel('Feature Contribution')
                axes[idx].grid(True, alpha=0.3)
                
                # Add prediction value
                pred = model.predict(sample)[0]
                axes[idx].text(0.02, 0.98, f'Prediction: {pred:.2f}', 
                             transform=axes[idx].transAxes, 
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                # Fallback: use feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                else:
                    importances = np.ones(len(feature_names)) / len(feature_names)
                
                top_features = np.argsort(importances)[-5:]
                features = [feature_names[i] for i in top_features]
                contributions = [importances[i] for i in top_features]
                
                colors = ['skyblue'] * len(features)
                bars = axes[idx].barh(range(len(features)), contributions, color=colors, alpha=0.7)
                axes[idx].set_yticks(range(len(features)))
                axes[idx].set_yticklabels(features)
                axes[idx].set_title(f'Sample {sample_idx} - Feature Importance')
                axes[idx].set_xlabel('Importance')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return {
            'type': 'image/png',
            'data': img_base64,
            'title': 'Individual Prediction Explanations',
            'description': 'Shows how each feature contributes to predictions for specific samples.'
        }
        
    except Exception as e:
        print(f"Error in generate_individual_predictions: {e}")
        return None
# --- END Enhanced XAI Methods ---

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True) 