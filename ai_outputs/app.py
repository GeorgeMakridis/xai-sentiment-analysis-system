from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import base64
import io
import numpy as np
from datetime import datetime
import re
from openai import OpenAI
from typing import List, Dict, Any, Optional
import hashlib
import uuid
import logging
import pickle
from faithfulness_evaluator import FaithfulnessEvaluator
from test_set_generator import TestSetGenerator
from plot_metadata_schema import build_plot_metadata
from vector_store import index_plot, index_plots_batch, rehydrate_from_restfs
from storage_abstraction import (
    save_plot_html,
    load_plot_html,
    load_registry,
    save_registry,
    delete_plot_file,
    save_plot_image,
    load_plot_image,
    load_plot_image_b64,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
SHARED_DATA_DIR = '/app/shared_data'
RESULTS_FOLDER = os.path.join(SHARED_DATA_DIR, 'results')
IMAGES_FOLDER = os.path.join(SHARED_DATA_DIR, 'images')
VECTOR_DB_FOLDER = os.path.join(SHARED_DATA_DIR, 'vector_db')

# Ensure directories exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY and OPENAI_API_KEY != 'your-openai-api-key':
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI API key configured successfully.")
else:
    logger.warning("OpenAI API key not set. Using fallback responses.")
    openai_client = None

# Global storage for vector database (in production, use a proper vector DB like Pinecone, Weaviate, etc.)
user_vector_db = {}

# Store conversation history for each user
conversation_history = {}

class VectorDatabase:
    """Simple in-memory vector database with user isolation"""
    
    def __init__(self):
        self.collections = {}
    
    def create_user_collection(self, user_id: str):
        """Create a new collection for a user"""
        if user_id not in self.collections:
            self.collections[user_id] = {
                'documents': [],
                'embeddings': [],
                'metadata': []
            }
    
    def add_document(self, user_id: str, text: str, metadata: Dict[str, Any], embedding: List[float] = None):
        """Add a document to user's collection"""
        if user_id not in self.collections:
            self.create_user_collection(user_id)
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = self._get_embedding(text)
        else:
            embedding = embedding
        
        self.collections[user_id]['documents'].append(text)
        self.collections[user_id]['embeddings'].append(embedding)
        self.collections[user_id]['metadata'].append(metadata)
    
    def has_documents(self, user_id: str) -> bool:
        """Check if user has any documents in the vector DB."""
        return user_id in self.collections and len(self.collections[user_id]['documents']) > 0

    def search(self, user_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if user_id not in self.collections:
            return []
        
        query_embedding = self._get_embedding(query)
        embeddings = self.collections[user_id]['embeddings']
        documents = self.collections[user_id]['documents']
        metadata = self.collections[user_id]['metadata']
        
        # Calculate cosine similarity
        similarities = []
        for i, doc_embedding in enumerate(embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity and return top_k results
        similarities.sort(reverse=True)
        results = []
        for similarity, idx in similarities[:top_k]:
            results.append({
                'text': documents[idx],
                'metadata': metadata[idx],
                'similarity': similarity
            })
        
        return results
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        if openai_client is None:
            # Fallback: simple hash-based embedding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            # Convert to list of floats
            embedding = [float(b) / 255.0 for b in hash_bytes] * 60  # Repeat to get 1536 dimensions
            return embedding[:1536]
        
        try:
            response = openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("Error getting embedding: %s", e)
            # Return a dummy embedding if OpenAI fails
            return [0.0] * 1536
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

# Initialize vector database
vector_db = VectorDatabase()

def save_image_to_shared_volume(image_data: str, user_id: str, image_type: str) -> Optional[str]:
    """Save base64 image to shared volume and return file path"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Create user-specific folder
        user_images_folder = os.path.join(IMAGES_FOLDER, user_id)
        os.makedirs(user_images_folder, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_type}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
        file_path = os.path.join(user_images_folder, filename)
        
        # Save image
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        return file_path
    except Exception as e:
        logger.error("Error saving image: %s", e)
        return None

def create_analysis_context(results: Dict[str, Any], user_id: str) -> List[str]:
    """Create context documents from analysis results for vector database"""
    context_docs = []
    
    # Model information
    if 'model_info' in results:
        model_info = results['model_info']
        model_doc = f"Model Type: {model_info.get('model_type', 'Unknown')}. "
        if 'feature_names' in model_info:
            features = ', '.join(model_info['feature_names'])
            model_doc += f"Features: {features}. "
        if 'analyzed_at' in model_info:
            model_doc += f"Analyzed at: {model_info['analyzed_at']}."
        
        context_docs.append(model_doc)
    
    # Performance metrics
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        metrics_doc = "Performance Metrics: "
        for key, value in metrics.items():
            metrics_doc += f"{key}: {value}, "
        context_docs.append(metrics_doc.rstrip(', '))
    
    # Data summary
    if 'data_summary' in results:
        data_summary = results['data_summary']
        data_doc = f"Dataset: Shape {data_summary.get('shape', 'Unknown')}. "
        if 'columns' in data_summary:
            columns = ', '.join(data_summary['columns'])
            data_doc += f"Columns: {columns}."
        context_docs.append(data_doc)
    
    # Feature importance
    if 'feature_importance' in results:
        importance = results['feature_importance']
        if isinstance(importance, dict):
            importance_doc = "Feature Importance: "
            for feature, score in importance.items():
                importance_doc += f"{feature}: {score}, "
            context_docs.append(importance_doc.rstrip(', '))
    
    # LIME Analysis Information
    lime_doc = "LIME Analysis: The analysis includes LIME (Local Interpretable Model-agnostic Explanations) visualizations that show local feature importance for individual predictions. LIME helps understand which words or features are most important for specific sentiment predictions."
    context_docs.append(lime_doc)
    
    # Visualization Summary
    if 'images' in results:
        viz_count = len(results['images'])
        viz_doc = f"Visualizations: The analysis generated {viz_count} comprehensive visualizations including LIME explanations, attention analysis plots, feature importance charts, correlation heatmaps, performance metrics, and detailed model explanations."
        context_docs.append(viz_doc)
        
        # Add specific visualization types
        viz_types = []
        if viz_count >= 6:  # Time series models
            viz_types.extend([
                "Enhanced Feature Importance Plot with color-coded bars and value labels",
                "Time Series Predictions with Confidence Intervals",
                "Feature Correlation Matrix with masked upper triangle",
                "Feature Distribution Analysis with 6-panel plots",
                "Model Performance Metrics visualization",
                "Time Series Decomposition (trend, seasonality, residuals)",
                "SHAP Summary Plot showing feature contributions",
                "SHAP Dependence Plots for top features",
                "SHAP Force Plots for sample predictions",
                "SHAP Waterfall Plot for individual predictions"
            ])
        elif viz_count >= 7:  # Text/sentiment models
            viz_types.extend([
                "Enhanced Word Importance Plot with top 25 words",
                "Multi-Class Sentiment Distributions",
                "Classification Performance Metrics",
                "Word Frequency Analysis",
                "Sentiment Trend Analysis",
                "Confusion Matrix visualization",
                "Word Cloud Analysis",
                "LIME Explanation for individual predictions",
                "Attention Analysis for transformer models",
                "Word Sentiment Association plots",
                "Feature Importance charts",
                "Model Performance visualizations"
            ])
        
        if viz_types:
            viz_details = "Detailed Visualizations: " + ". ".join(viz_types) + "."
            context_docs.append(viz_details)
    
    # LIME analysis summary
    if 'lime_analysis' in results and results['lime_analysis']:
        lime_analysis = results['lime_analysis']
        if 'top_features' in lime_analysis and lime_analysis['top_features']:
            top_feats = lime_analysis['top_features']
            lime_doc = "Top LIME Features: " + ", ".join([f"{f['feature']} (importance={f['importance']:.4f})" for f in top_feats])
            context_docs.append(lime_doc)
            
            # Add detailed LIME analysis
            lime_details = f"LIME Analysis Details: Analyzed {lime_analysis.get('total_features_analyzed', 0)} features for local explanations. "
            lime_details += f"Top feature '{top_feats[0]['feature']}' has the highest importance ({top_feats[0]['importance']:.4f}), indicating it has the strongest influence on this specific prediction."
            context_docs.append(lime_details)
        elif 'error' in lime_analysis:
            lime_doc = f"LIME Analysis Error: {lime_analysis['error']}"
            context_docs.append(lime_doc)
    
    # Visualization descriptions
    if 'images' in results:
        viz_descriptions = [
            "Feature Importance Analysis: Shows which variables have the most impact on model predictions",
            "SHAP Summary Plot: Displays how each feature contributes to individual predictions",
            "Feature Correlation Heatmap: Shows relationships between different features",
            "Feature Distributions: Displays the spread and shape of feature values",
            "Model Performance Analysis: Shows accuracy, precision, recall, and other metrics",
            "Model Explainability Summary: Provides overall insights into model behavior"
        ]
        
        for i, description in enumerate(viz_descriptions):
            if i < len(results['images']):
                context_docs.append(description)
    
    return context_docs

def save_results_to_shared_volume(results: Dict[str, Any], user_id: str) -> Optional[str]:
    """Save analysis results as JSON file to shared volume"""
    try:
        # Create user-specific results folder
        user_results_folder = os.path.join(RESULTS_FOLDER, user_id)
        os.makedirs(user_results_folder, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.json"
        file_path = os.path.join(user_results_folder, filename)
        
        # Save results as JSON
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Saved results to %s", file_path)
        return file_path
    except Exception as e:
        logger.error("Error saving results to shared volume: %s", e)
        return None

def store_results_in_vector_db(results: Dict[str, Any], user_id: str):
    """Store analysis results in vector database"""
    try:
        # Save results to shared volume first
        save_results_to_shared_volume(results, user_id)

        # Handle different types of results
        result_type = results.get('type', 'unknown')

        if 'plot_summaries' in results and isinstance(results['plot_summaries'], list):
            for summary in results['plot_summaries']:
                if isinstance(summary, dict):
                    index_plot(vector_db, user_id, summary, source='results_plot_summaries')
        
        # Check for data_statistics in the new format
        if 'data_statistics' in results:
            data_stats = results['data_statistics']
            data_type = data_stats.get('data_type', 'unknown')
            visualizations = data_stats.get('visualizations', [])
            
            data_doc = f"Data Statistics Analysis: Data type: {data_type}. "
            data_doc += "Analysis includes: comprehensive data overview, sentiment distribution histogram, per-asset sentiment boxplots, keyword frequency analysis, word sentiment associations, asset distribution charts, and text length distribution. "
            data_doc += "Data Overview: Shows dataset statistics including total articles, average title length, date range, distinct assets, and column information. "
            data_doc += "Sentiment Distribution: Histogram showing the distribution of sentiment scores across all articles with neutral (0) as reference point. "
            data_doc += "Per-Asset Sentiment: Boxplot analysis showing sentiment patterns for the top 25 assets by article count. "
            data_doc += "Keyword Insights: Horizontal bar chart showing the top 15 most frequent words in article titles. "
            data_doc += "Word Sentiment Associations: Two-panel chart showing top 10 words driving positive sentiment (green bars) and negative sentiment (red bars) with exact scores. "
            data_doc += "Asset Distribution: Horizontal bar chart showing the top 20 assets by article count. "
            data_doc += "Text Length Distribution: Histogram showing the distribution of article title lengths with mean, median, and standard deviation markers. "
            data_doc += f" Visualizations generated: {', '.join(visualizations)}."
            
            # Add actual plot data to the document
            plot_data = data_stats.get('plot_data', {})
            if 'word_sentiment' in plot_data:
                word_sentiment = plot_data['word_sentiment']
                logger.debug("Received plot_data keys: %s; word_sentiment positive=%s negative=%s",
                            list(plot_data.keys()),
                            bool(word_sentiment.get('positive_words')),
                            bool(word_sentiment.get('negative_words')))
                pos_words = word_sentiment.get('positive_words', [])
                neg_words = word_sentiment.get('negative_words', [])
                
                if pos_words:
                    data_doc += f" Top 10 positive words: {', '.join([f'{word}({score:.2f})' for word, score in pos_words[:5]])}. "
                if neg_words:
                    data_doc += f" Top 10 negative words: {', '.join([f'{word}({score:.2f})' for word, score in neg_words[:5]])}. "
                index_plot(vector_db, user_id, {
                    'title': 'Word Sentiment Associations',
                    'plot_type': 'word_sentiment_association',
                    'description': 'Top words driving positive and negative sentiment.',
                    'data': word_sentiment,
                    'summary_text': 'Top positive and negative sentiment-associated words.'
                }, source='data_statistics_plot_data')
            
            if 'keywords' in plot_data:
                keywords = plot_data['keywords'].get('top_keywords', [])
                if keywords:
                    data_doc += f" Top 15 keywords: {', '.join([f'{word}({count})' for word, count in keywords[:5]])}. "
                index_plot(vector_db, user_id, {
                    'title': 'Top Keywords',
                    'plot_type': 'keyword_frequency',
                    'description': 'Most frequent words in titles.',
                    'data': plot_data.get('keywords', {}),
                    'summary_text': 'Most frequent keywords in the dataset.'
                }, source='data_statistics_plot_data')
            
            # Also store plot_summaries if they're included directly in data_statistics
            if 'plot_summaries' in data_stats and isinstance(data_stats['plot_summaries'], list):
                for summary in data_stats['plot_summaries']:
                    if isinstance(summary, dict):
                        index_plot(vector_db, user_id, summary, source='data_statistics_plot_summaries')
            
            metadata = {
                'user_id': user_id,
                'doc_type': 'data_statistics',
                'data_type': data_type,
                'visualizations': visualizations,
                'timestamp': data_stats.get('timestamp', datetime.now().isoformat())
            }
            vector_db.add_document(user_id, data_doc, metadata)
            
            # Store images and their metadata
            if 'images' in results:
                for i, image_data in enumerate(results['images']):
                    image_type = f"data_stats_{i+1}"
                    
                    # Handle both base64 and file path formats
                    if isinstance(image_data, dict) and 'image' in image_data:
                        # Base64 image data
                        base64_data = image_data['image']
                        image_path = save_image_to_shared_volume(base64_data, user_id, image_type)
                        
                        if image_path:
                            # Create image metadata document
                            image_doc = f"Data Statistics Visualization {i+1}: {image_type} stored at {image_path}"
                            metadata = {
                                'user_id': user_id,
                                'doc_type': 'data_statistics_visualization',
                                'image_path': image_path,
                                'image_type': image_type,
                                'index': i
                            }
                            vector_db.add_document(user_id, image_doc, metadata)
                        if isinstance(image_data, dict) and image_data.get('summary'):
                            summary_payload = {
                                'title': image_data.get('title', f"Data Statistics Visualization {i+1}"),
                                'plot_type': image_data.get('type', image_type),
                                'description': image_data.get('description', ''),
                                'data': image_data.get('data', {}),
                                'metadata': image_data.get('metadata', {}),
                                'summary_text': image_data.get('summary')
                            }
                            index_plot(vector_db, user_id, summary_payload, source='data_statistics_images')
                        if isinstance(image_data, dict) and image_data.get('summary'):
                            summary_payload = {
                                'title': image_data.get('title', f"Data Statistics Visualization {i+1}"),
                                'plot_type': image_data.get('type', image_type),
                                'description': image_data.get('description', ''),
                                'data': image_data.get('data', {}),
                                'metadata': image_data.get('metadata', {}),
                                'summary_text': image_data.get('summary')
                            }
                            index_plot(vector_db, user_id, summary_payload, source='data_statistics_images')
                    elif isinstance(image_data, str):
                        # Direct base64 string
                        image_path = save_image_to_shared_volume(image_data, user_id, image_type)
                        
                        if image_path:
                            # Create image metadata document
                            image_doc = f"Data Statistics Visualization {i+1}: {image_type} stored at {image_path}"
                            metadata = {
                                'user_id': user_id,
                                'doc_type': 'data_statistics_visualization',
                                'image_path': image_path,
                                'image_type': image_type,
                                'index': i
                            }
                            vector_db.add_document(user_id, image_doc, metadata)
                        if isinstance(image_data, dict) and image_data.get('summary'):
                            summary_payload = {
                                'title': image_data.get('title', f"Data Statistics Visualization {i+1}"),
                                'plot_type': image_data.get('type', image_type),
                                'description': image_data.get('description', ''),
                                'data': image_data.get('data', {}),
                                'metadata': image_data.get('metadata', {}),
                                'summary_text': image_data.get('summary')
                            }
                            index_plot(vector_db, user_id, summary_payload, source='data_statistics_images')
                        if isinstance(image_data, dict) and image_data.get('summary'):
                            summary_payload = {
                                'title': image_data.get('title', f"Data Statistics Visualization {i+1}"),
                                'plot_type': image_data.get('type', image_type),
                                'description': image_data.get('description', ''),
                                'data': image_data.get('data', {}),
                                'metadata': image_data.get('metadata', {}),
                                'summary_text': image_data.get('summary')
                            }
                            index_plot(vector_db, user_id, summary_payload, source='data_statistics_images')
        
        elif result_type == 'data_statistics':
            # Store data statistics results
            data_doc = f"Data Statistics Analysis: Data type: {results.get('data_type', 'unknown')}. "
            data_doc += "Analysis includes: word sentiment associations, keyword insights, sentiment distribution, per-asset sentiment analysis, and comprehensive data overview. "
            data_doc += "Word sentiment analysis shows the top 10 words driving positive and negative sentiment in the dataset. "
            data_doc += "Keyword insights show the most frequent words in article titles. "
            data_doc += "Asset-specific analysis shows sentiment patterns by financial asset/ticker."
            
            metadata = {
                'user_id': user_id,
                'doc_type': 'data_statistics',
                'data_type': results.get('data_type', 'unknown'),
                'insights': results.get('insights', {}),
                'timestamp': results.get('timestamp', datetime.now().isoformat())
            }
            vector_db.add_document(user_id, data_doc, metadata)

            plot_data = results.get('plot_data', {})
            if 'word_sentiment' in plot_data:
                index_plot(vector_db, user_id, {
                    'title': 'Word Sentiment Associations',
                    'plot_type': 'word_sentiment_association',
                    'description': 'Top words driving positive and negative sentiment.',
                    'data': plot_data.get('word_sentiment', {}),
                    'summary_text': 'Top positive and negative sentiment-associated words.'
                }, source='data_statistics_plot_data')
            if 'keywords' in plot_data:
                index_plot(vector_db, user_id, {
                    'title': 'Top Keywords',
                    'plot_type': 'keyword_frequency',
                    'description': 'Most frequent words in titles.',
                    'data': plot_data.get('keywords', {}),
                    'summary_text': 'Most frequent keywords in the dataset.'
                }, source='data_statistics_plot_data')
            
            # Store images and their metadata
            if 'images' in results:
                for i, image_data in enumerate(results['images']):
                    image_type = f"data_stats_{i+1}"
                    
                    # Handle both base64 and file path formats
                    if isinstance(image_data, dict) and 'image' in image_data:
                        # Base64 image data
                        base64_data = image_data['image']
                        image_path = save_image_to_shared_volume(base64_data, user_id, image_type)
                        
                        if image_path:
                            # Create image metadata document
                            image_doc = f"Data Statistics Visualization {i+1}: {image_type} stored at {image_path}"
                            metadata = {
                                'user_id': user_id,
                                'doc_type': 'data_statistics_visualization',
                                'image_path': image_path,
                                'image_type': image_type,
                                'index': i
                            }
                            vector_db.add_document(user_id, image_doc, metadata)
                    elif isinstance(image_data, str):
                        # Direct base64 string
                        image_path = save_image_to_shared_volume(image_data, user_id, image_type)
                        
                        if image_path:
                            # Create image metadata document
                            image_doc = f"Data Statistics Visualization {i+1}: {image_type} stored at {image_path}"
                            metadata = {
                                'user_id': user_id,
                                'doc_type': 'data_statistics_visualization',
                                'image_path': image_path,
                                'image_type': image_type,
                                'index': i
                            }
                            vector_db.add_document(user_id, image_doc, metadata)
        
        elif result_type == 'xai_analysis':
            # Clear previous XAI results for this user
            if user_id in vector_db.collections:
                # Remove old XAI-related documents
                old_docs = []
                old_embeddings = []
                old_metadata = []
                
                for i, metadata_doc in enumerate(vector_db.collections[user_id]['metadata']):
                    if metadata_doc.get('doc_type') in ['xai_analysis', 'xai_visualization']:
                        # Skip this document (don't add to new lists)
                        continue
                    else:
                        # Keep this document
                        old_docs.append(vector_db.collections[user_id]['documents'][i])
                        old_embeddings.append(vector_db.collections[user_id]['embeddings'][i])
                        old_metadata.append(metadata_doc)
                
                # Replace the collection with only non-XAI documents
                vector_db.collections[user_id]['documents'] = old_docs
                vector_db.collections[user_id]['embeddings'] = old_embeddings
                vector_db.collections[user_id]['metadata'] = old_metadata
            
            # Store new XAI analysis results
            xai_analysis = results.get('xai_analysis', {})
            xai_data = xai_analysis.get('visualizations', results.get('visualizations', {}))
            example_index = xai_analysis.get('example_index', results.get('example_index', 'N/A'))
            model_type = xai_analysis.get('model_type', results.get('model_type', 'N/A'))
            
            # Extract structured XAI artifacts for faithfulness evaluation
            lime_features = xai_analysis.get('lime_features', [])
            attention_tokens = xai_analysis.get('attention_tokens', [])
            confidence_score = xai_analysis.get('confidence_score', None)
            
            xai_doc = f"XAI Analysis Results: Example index: {example_index}. "
            xai_doc += f"Model type: {model_type}. "
            xai_doc += f"Visualizations generated: {', '.join(xai_data.keys()) if isinstance(xai_data, dict) else 'multiple visualizations'}. "
            
            # Add LIME features to document text
            if lime_features:
                lime_features_str = ", ".join([f"'{word}' (importance={score:.3f})" for word, score in lime_features[:10]])
                xai_doc += f"LIME top features: {lime_features_str}. "
            
            # Add attention tokens to document text
            if attention_tokens:
                attention_tokens_str = ", ".join([f"'{token}' (score={score:.3f})" for token, score in attention_tokens[:10]])
                xai_doc += f"Attention top tokens: {attention_tokens_str}. "
            
            # Add confidence score to document text
            if confidence_score is not None:
                xai_doc += f"Prediction confidence: {confidence_score:.3f}. "
            
            # Add specific descriptions for each visualization type
            if 'lime' in xai_data and xai_data['lime']:
                xai_doc += "LIME Analysis: Local Interpretable Model-agnostic Explanations showing which words are most important for the specific sentiment prediction. "
                xai_doc += "LIME highlights the top contributing words that drive the model's decision for this particular text example. "
            
            if 'attention' in xai_data and xai_data['attention']:
                xai_doc += "Attention Analysis: Shows attention weights from the transformer model, highlighting which words the model focuses on when making predictions. "
            
            if 'confidence' in xai_data and xai_data['confidence']:
                xai_doc += "Prediction Confidence: Visualizes the confidence levels of the model's predictions across different sentiment categories. "
            
            metadata = {
                'user_id': user_id,
                'doc_type': 'xai_analysis',
                'example_index': example_index,
                'model_type': model_type,
                'visualizations': list(xai_data.keys()) if isinstance(xai_data, dict) else [],
                'timestamp': results.get('timestamp', datetime.now().isoformat()),
                # Structured XAI artifacts for faithfulness evaluation
                'lime_features': lime_features,  # List of (word, score) tuples
                'attention_tokens': attention_tokens,  # List of (token, score) tuples
                'confidence_score': confidence_score,  # Float confidence value
                'example_text': xai_analysis.get('example_text', '')
            }
            vector_db.add_document(user_id, xai_doc, metadata)
            
            # Store XAI visualizations
            if isinstance(xai_data, dict):
                for viz_type, viz_data in xai_data.items():
                    if isinstance(viz_data, str):  # Base64 image
                        image_path = save_image_to_shared_volume(viz_data, user_id, f"xai_{viz_type}")
                        if image_path:
                            viz_doc = f"XAI Visualization: {viz_type} stored at {image_path}"
                            metadata = {
                                'user_id': user_id,
                                'doc_type': 'xai_visualization',
                                'visualization_type': viz_type,
                                'image_path': image_path,
                                'example_index': example_index
                            }
                            vector_db.add_document(user_id, viz_doc, metadata)
                    if isinstance(viz_data, dict) and viz_data.get('summary'):
                        summary_payload = {
                            'title': viz_data.get('title', f"XAI Visualization {viz_type}"),
                            'plot_type': viz_type,
                            'description': viz_data.get('description', ''),
                            'data': viz_data.get('data', {}),
                            'metadata': viz_data.get('metadata', {}),
                            'summary_text': viz_data.get('summary')
                        }
                        index_plot(vector_db, user_id, summary_payload, source='xai_visualizations')
        
        else:
            # Handle legacy results format
            # Create context documents
            context_docs = create_analysis_context(results, user_id)
            
            # Store each document in vector database
            for i, doc in enumerate(context_docs):
                metadata = {
                    'user_id': user_id,
                    'doc_type': 'analysis_context',
                    'index': i,
                    'timestamp': datetime.now().isoformat()
                }
                vector_db.add_document(user_id, doc, metadata)
            
            # Store images and their metadata
            if 'images' in results:
                for i, image_data in enumerate(results['images']):
                    image_type = f"viz_{i+1}"
                    image_path = save_image_to_shared_volume(image_data, user_id, image_type)
                    
                    if image_path:
                        # Create image metadata document
                        image_doc = f"Visualization {i+1}: {image_type} stored at {image_path}"
                        metadata = {
                            'user_id': user_id,
                            'doc_type': 'visualization',
                            'image_path': image_path,
                            'image_type': image_type,
                            'index': i
                        }
                        vector_db.add_document(user_id, image_doc, metadata)
                    if isinstance(image_data, dict) and image_data.get('summary'):
                        summary_payload = {
                            'title': image_data.get('title', f"Visualization {i+1}"),
                            'plot_type': image_data.get('type', image_type),
                            'description': image_data.get('description', ''),
                            'data': image_data.get('data', {}),
                            'metadata': image_data.get('metadata', {}),
                            'summary_text': image_data.get('summary')
                        }
                        index_plot(vector_db, user_id, summary_payload, source='results_images')
        
        logger.info("Stored results in vector database for user %s, type: %s", user_id, result_type)
        
    except Exception as e:
        logger.error("Error storing results in vector database: %s", e)

def add_to_conversation_history(user_id: str, question: str, answer: str):
    """Add a question-answer pair to user's conversation history"""
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    conversation_history[user_id].append({
        'question': question,
        'answer': answer,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 10 exchanges to prevent context from getting too long
    if len(conversation_history[user_id]) > 10:
        conversation_history[user_id] = conversation_history[user_id][-10:]

def get_conversation_context(user_id: str) -> str:
    """Get recent conversation context for a user"""
    if user_id not in conversation_history or not conversation_history[user_id]:
        return ""
    
    context_parts = []
    for exchange in conversation_history[user_id][-5:]:  # Last 5 exchanges
        context_parts.append(f"Previous Q: {exchange['question']}")
        context_parts.append(f"Previous A: {exchange['answer']}")
    
    return "\n".join(context_parts)

def clear_conversation_history(user_id: str):
    """Clear conversation history for a user"""
    if user_id in conversation_history:
        del conversation_history[user_id]

def format_context_for_llm(relevant_docs: List[Dict[str, Any]], plot_summaries: List[Dict[str, Any]]) -> str:
    """Format context in a way that's easier for LLM to parse and use"""
    context_parts = []
    
    # Extract plot summaries with structured data
    if plot_summaries:
        context_parts.append("=== PLOT SUMMARIES AND DATA ===\n")
        for summary in plot_summaries:
            plot_type = summary.get('plot_type', 'unknown')
            title = summary.get('title', 'Untitled Plot')
            description = summary.get('description', '')
            data = summary.get('data', {})
            
            context_parts.append(f"\nPlot: {title} (Type: {plot_type})")
            context_parts.append(f"Description: {description}")
            
            # Format structured data clearly
            if isinstance(data, dict):
                if 'positive_words' in data:
                    pos_words = data['positive_words']
                    if pos_words:
                        context_parts.append(f"\nPositive Words (with sentiment scores):")
                        for word, score in pos_words[:15]:  # Show top 15
                            # Check if score is a sentiment score (typically -1 to 1) or a count
                            if isinstance(score, (int, float)):
                                if abs(score) > 10:
                                    context_parts.append(f"  - {word}: {int(score)} occurrences")
                                else:
                                    context_parts.append(f"  - {word}: {score:.3f} sentiment score")
                            else:
                                context_parts.append(f"  - {word}: {score}")
                
                if 'negative_words' in data:
                    neg_words = data['negative_words']
                    if neg_words:
                        context_parts.append(f"\nNegative Words (with sentiment scores):")
                        for word, score in neg_words[:15]:
                            if isinstance(score, (int, float)):
                                if abs(score) > 10:
                                    context_parts.append(f"  - {word}: {int(score)} occurrences")
                                else:
                                    context_parts.append(f"  - {word}: {score:.3f} sentiment score")
                            else:
                                context_parts.append(f"  - {word}: {score}")
                
                if 'top_keywords' in data:
                    keywords = data['top_keywords']
                    if keywords:
                        context_parts.append(f"\nTop Keywords (with frequencies):")
                        for keyword, freq in keywords[:15]:
                            context_parts.append(f"  - {keyword}: {freq} occurrences")
                
                # Handle image analysis data
                if 'class_distribution' in data:
                    class_dist = data['class_distribution']
                    if isinstance(class_dist, dict):
                        context_parts.append(f"\nClass Distribution:")
                        for class_name, count in list(class_dist.items())[:15]:
                            context_parts.append(f"  - {class_name}: {count} samples")
                elif 'top_classes' in data:
                    # Fallback: use top_classes if class_distribution not available
                    class_dist = data['top_classes']
                    if isinstance(class_dist, dict):
                        context_parts.append(f"\nClass Distribution:")
                        for class_name, count in list(class_dist.items())[:15]:
                            context_parts.append(f"  - {class_name}: {count} samples")
                
                if 'confusion_matrix' in data:
                    context_parts.append(f"\nConfusion Matrix: Available in plot visualization")
            
            context_parts.append("\n")
    
    # Add other relevant documents (excluding plot summaries already included)
    if relevant_docs:
        context_parts.append("\n=== ADDITIONAL ANALYSIS CONTEXT ===\n")
        for doc in relevant_docs:
            if doc.get('metadata', {}).get('doc_type') != 'plot_summary':  # Skip if already in plot summaries
                context_parts.append(doc['text'])
                context_parts.append("\n")
    
    return "\n".join(context_parts)

def generate_structured_fallback_response(question: str, relevant_docs: List[Dict[str, Any]], 
                                         plot_summaries: List[Dict[str, Any]], 
                                         conversation_context: str = "", user_id: str = None) -> str:
    """Generate structured response when LLM is unavailable"""
    question_lower = question.lower()
    
    # Count ALL visualizations from user's collection in vector DB
    total_plots = 0
    if user_id and user_id in vector_db.collections:
        # Count all visualization documents
        for metadata in vector_db.collections[user_id]['metadata']:
            doc_type = metadata.get('doc_type', '')
            if doc_type in ['plot_summary', 'data_statistics_visualization', 'xai_visualization']:
                total_plots += 1
    else:
        # Fallback: count plot summaries if user_id not available
        total_plots = len(plot_summaries)
        # Also count data statistics visualizations from relevant docs
        for doc in relevant_docs:
            metadata = doc.get('metadata', {})
            if metadata.get('doc_type') == 'data_statistics_visualization':
                total_plots += 1
    
    # Check question type to provide relevant response
    is_about_words = any(word in question_lower for word in ["word", "words", "check for", "should i check", "what words"])
    is_about_sentiment = any(word in question_lower for word in ["sentiment", "distribution", "positive", "negative"])
    is_about_keywords = any(word in question_lower for word in ["keyword", "keywords", "topics", "main topics"])
    is_about_visualizations = any(word in question_lower for word in ["visualization", "plot", "chart", "graph"])
    
    # Collect all words from all plot summaries
    all_positive_words = []
    all_negative_words = []
    all_keywords = []
    
    for summary in plot_summaries:
        data = summary.get('data', {})
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except:
                data = {}
        
        if isinstance(data, dict):
            pos_words = data.get('positive_words', [])
            neg_words = data.get('negative_words', [])
            keywords = data.get('top_keywords', [])
            
            if pos_words:
                if isinstance(pos_words[0], (list, tuple)) and len(pos_words[0]) == 2:
                    all_positive_words.extend([w for w, s in pos_words])
                else:
                    all_positive_words.extend([str(w) for w in pos_words])
            
            if neg_words:
                if isinstance(neg_words[0], (list, tuple)) and len(neg_words[0]) == 2:
                    all_negative_words.extend([w for w, s in neg_words])
                else:
                    all_negative_words.extend([str(w) for w in neg_words])
            
            if keywords:
                if isinstance(keywords[0], (list, tuple)) and len(keywords[0]) == 2:
                    all_keywords.extend([k for k, f in keywords])
                else:
                    all_keywords.extend([str(k) for k in keywords])
    
    # Remove duplicates while preserving order
    unique_positive = list(dict.fromkeys(all_positive_words))[:15]
    unique_negative = list(dict.fromkeys(all_negative_words))[:15]
    unique_keywords = list(dict.fromkeys(all_keywords))[:15]
    
    # Handle questions about words to check
    if is_about_words:
        response_parts = []
        if unique_positive or unique_negative:
            response_parts.append("Based on sentiment analysis, here are the key words you should check:")
            if unique_positive:
                pos_list = ", ".join(unique_positive)
                response_parts.append(f"\nPositive sentiment words: {pos_list}")
            if unique_negative:
                neg_list = ", ".join(unique_negative)
                response_parts.append(f"\nNegative sentiment words: {neg_list}")
            if unique_keywords:
                kw_list = ", ".join(unique_keywords)
                response_parts.append(f"\nMost frequent keywords: {kw_list}")
            return "\n".join(response_parts)
    
    # Handle questions about sentiment distribution
    if is_about_sentiment and not is_about_words:
        response_parts = []
        if unique_positive or unique_negative:
            response_parts.append(f"Sentiment distribution analysis (from {total_plots} visualization(s)):")
            if unique_positive:
                pos_list = ", ".join(unique_positive[:10])
                response_parts.append(f"\nTop positive sentiment drivers: {pos_list}")
            if unique_negative:
                neg_list = ", ".join(unique_negative[:10])
                response_parts.append(f"\nTop negative sentiment drivers: {neg_list}")
            response_parts.append("\nInsight: These words show the main sentiment patterns in your data. Positive words indicate favorable content, while negative words suggest areas that may need attention.")
            return "\n".join(response_parts)
    
    # Handle questions about keywords
    if is_about_keywords:
        if unique_keywords:
            kw_list = ", ".join(unique_keywords)
            return f"Main topics and keywords in your dataset: {kw_list}\n\nThese represent the most frequently discussed subjects, indicating the primary focus areas of your data."
    
    # Handle general visualization questions
    if is_about_visualizations:
        response_parts = [f"You have {total_plots} visualization(s) available:"]
        seen_types = set()
        for summary in plot_summaries:
            plot_type = summary.get('plot_type', 'unknown')
            title = summary.get('title', 'Untitled')
            if plot_type not in seen_types:
                response_parts.append(f"- {title} ({plot_type})")
                seen_types.add(plot_type)
        return "\n".join(response_parts)
    
    # Default response - provide comprehensive overview
    if plot_summaries:
        response_parts = [f"Based on your {total_plots} visualization(s), here are key insights:"]
        
        if unique_positive:
            pos_list = ", ".join(unique_positive[:10])
            response_parts.append(f"\nPositive sentiment drivers: {pos_list}")
            response_parts.append("Insight: These words are associated with positive sentiment. Consider focusing content around these themes.")
        
        if unique_negative:
            neg_list = ", ".join(unique_negative[:10])
            response_parts.append(f"\nNegative sentiment drivers: {neg_list}")
            response_parts.append("Insight: These words correlate with negative sentiment. Monitor or address concerns related to these topics.")
        
        if unique_keywords:
            kw_list = ", ".join(unique_keywords[:10])
            response_parts.append(f"\nMain topics: {kw_list}")
            response_parts.append("Insight: These are the most frequently discussed topics, indicating the primary focus areas.")
        
        if not unique_positive and not unique_negative and not unique_keywords:
            response_parts.append(f"\nI can see {total_plots} visualization(s) in your data. Ask specific questions to explore insights.")
        
        return "\n".join(response_parts)
    else:
        # Even without plot summaries, try to extract insights from relevant docs
        if relevant_docs:
            response_parts = ["Based on your analysis results:"]
            for doc in relevant_docs[:3]:
                doc_text = doc.get('text', '')
                if doc_text and len(doc_text) > 20:
                    if 'dataset' in doc_text.lower() or 'shape' in doc_text.lower():
                        response_parts.append(f"\nDataset Information: {doc_text[:200]}...")
                    elif 'model' in doc_text.lower():
                        response_parts.append(f"\nModel Information: {doc_text[:200]}...")
            
            response_parts.append("\nTo get deeper insights:")
            response_parts.append("- Ask 'What words should I check for?' for sentiment words")
            response_parts.append("- Ask 'What patterns do you see?' for trend analysis")
            response_parts.append("- Ask 'How should I clean this data?' for data quality recommendations")
            return "\n".join(response_parts)
        else:
            return "I can see you have analysis results available. Please ask specific questions about your data, model, or visualizations. For example: 'What words should I check for?' or 'What patterns do you see in the data?'"

def extract_xai_artifacts_from_docs(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract structured XAI artifacts from retrieved documents
    
    Args:
        docs: List of documents from vector DB search
        
    Returns:
        Dictionary with lime_features, attention_tokens, confidence_score
    """
    artifacts = {
        'lime_features': [],
        'attention_tokens': [],
        'confidence_score': None
    }
    
    for doc in docs:
        metadata = doc.get('metadata', {})
        
        # Extract LIME features
        if 'lime_features' in metadata and metadata['lime_features']:
            artifacts['lime_features'] = metadata['lime_features']
        
        # Extract attention tokens
        if 'attention_tokens' in metadata and metadata['attention_tokens']:
            artifacts['attention_tokens'] = metadata['attention_tokens']
        
        # Extract confidence score
        if 'confidence_score' in metadata and metadata['confidence_score'] is not None:
            artifacts['confidence_score'] = metadata['confidence_score']
    
    return artifacts


def generate_constrained_prompt(question: str, 
                               formatted_context: str,
                               xai_artifacts: Dict[str, Any],
                               conversation_context: str = "") -> tuple:
    """
    Generate constrained prompt with explicit grounding requirements
    
    Args:
        question: User question
        formatted_context: Formatted context from vector DB
        xai_artifacts: Dictionary with lime_features, attention_tokens, confidence_score
        conversation_context: Previous conversation history
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Build artifact lists for prompt
    lime_list = []
    if xai_artifacts.get('lime_features'):
        lime_list = [f"'{word}' (importance={score:.3f})" 
                     for word, score in xai_artifacts['lime_features'][:10]]
    
    attention_list = []
    if xai_artifacts.get('attention_tokens'):
        attention_list = [f"'{token}' (score={score:.3f})" 
                          for token, score in xai_artifacts['attention_tokens'][:10]]
    
    confidence_val = xai_artifacts.get('confidence_score')
    
    system_prompt = """You are explaining a FinBERT sentiment prediction based on XAI (Explainable AI) analysis results.

CRITICAL CONSTRAINTS - YOU MUST FOLLOW THESE:

1. ONLY reference the following retrieved facts from XAI analysis:
"""
    
    if lime_list:
        system_prompt += f"   - LIME top features: {', '.join(lime_list)}\n"
    if attention_list:
        system_prompt += f"   - Attention top tokens: {', '.join(attention_list)}\n"
    if confidence_val is not None:
        system_prompt += f"   - Confidence score: {confidence_val:.3f}\n"
    
    system_prompt += """
2. DO NOT:
   - Invent features not in the lists above
   - Overstate confidence beyond the provided score
   - Claim causal relationships not supported by the data
   - Make up numbers or values not in the retrieved context

3. CITATION REQUIREMENTS:
   - For each claim, cite the source (e.g., "According to LIME analysis...", "Attention analysis shows...")
   - If information is missing, say "Information not available in analysis results"
   - Always reference where your information comes from

4. BE ACCURATE:
   - Only mention words/tokens that appear in the lists above
   - Only use confidence values that match the provided score
   - If you're not sure, say so rather than guessing

Your goal is to provide a faithful, grounded explanation that accurately reflects the XAI analysis results.

Do NOT use markdown formatting (no **, ##, ```, -, * bullets). Write in plain text with natural sentence structure."""

    user_prompt = f"""User Question: {question}

AVAILABLE DATA AND CONTEXT:
{formatted_context}

PREVIOUS CONVERSATION:
{conversation_context if conversation_context else "None"}

REMEMBER:
- Only reference features/tokens/values from the context above
- Cite your sources for each claim
- If information is missing, acknowledge it
- Be accurate and faithful to the XAI analysis results"""

    return system_prompt, user_prompt


def generate_naive_rag_response(question: str, user_id: str) -> str:
    """
    Generate RAG response using naive prompt (no constraints, baseline)
    
    Args:
        question: User question
        user_id: User identifier
        
    Returns:
        RAG-generated response
    """
    try:
        # Search for relevant context
        relevant_docs = vector_db.search(user_id, question, top_k=8)
        
        if not relevant_docs:
            return "I don't have access to your analysis results yet. Please upload and analyze a model first."
        
        # Extract plot summaries (same as constrained version)
        def extract_plot_summary_data(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            summaries = []
            for doc in docs:
                if doc.get('metadata', {}).get('doc_type') != 'plot_summary':
                    continue
                meta = doc.get('metadata', {})
                plot_type = meta.get('plot_type', '')
                data = meta.get('data')
                if not data and isinstance(doc.get('text'), str):
                    text = doc['text']
                    marker = "Data: "
                    if marker in text:
                        try:
                            data_str = text.split(marker, 1)[1]
                            if "Metadata:" in data_str:
                                data_str = data_str.split("Metadata:", 1)[0]
                            data_str = data_str.strip().rstrip(". ")
                            data = json.loads(data_str)
                        except Exception:
                            data = None
                summary_data = data or {}
                summaries.append({
                    'title': meta.get('title', ''),
                    'plot_type': plot_type,
                    'data': summary_data,
                    'description': meta.get('description', ''),
                    'summary_text': meta.get('summary_text', '')
                })
            return summaries
        
        plot_summaries = extract_plot_summary_data(relevant_docs)
        
        # Format context
        formatted_context = format_context_for_llm(relevant_docs, plot_summaries)
        conversation_context = get_conversation_context(user_id)
        
        # Naive system prompt (no constraints)
        system_prompt = """You are an AI assistant explaining sentiment analysis predictions. 
Explain the prediction based on the context provided. Be helpful and informative.
Do NOT use markdown formatting (no **, ##, ```, -, * bullets). Write in plain text with natural sentence structure."""

        # Naive user prompt (no constraints)
        user_prompt = f"""User Question: {question}

Context:
{formatted_context}

Previous Conversation:
{conversation_context if conversation_context else "None"}

Please explain the sentiment prediction based on the context above."""

        # Generate response
        if openai_client is None:
            return generate_structured_fallback_response(question, relevant_docs, plot_summaries, conversation_context, user_id)
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error with OpenAI API: %s", e)
            return generate_structured_fallback_response(question, relevant_docs, plot_summaries, conversation_context, user_id)
        
    except Exception as e:
        logger.error("Error generating naive RAG response: %s", e)
        return f"I encountered an error while processing your question: {str(e)}"


def generate_rag_response(question: str, user_id: str, use_constrained: bool = True) -> str:
    """Generate response using RAG approach"""
    try:
        # Search for relevant context
        relevant_docs = vector_db.search(user_id, question, top_k=8)  # Increased for richer context
        
        logger.debug("Found %d relevant documents for question: '%s'", len(relevant_docs), question)
        for i, doc in enumerate(relevant_docs):
            logger.debug("Doc %d: %s... (similarity: %.3f)", i+1, doc['text'][:100], doc['similarity'])
        
        if not relevant_docs:
            return "I don't have access to your analysis results yet. Please upload and analyze a model first."
        
        # Extract plot summaries from relevant documents
        def extract_plot_summary_data(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            import sys
            summaries = []
            for doc in docs:
                if doc.get('metadata', {}).get('doc_type') != 'plot_summary':
                    continue
                meta = doc.get('metadata', {})
                plot_type = meta.get('plot_type', '')
                data = meta.get('data')
                logger.debug("extract: plot_type=%s, data_type=%s, data_keys=%s",
                             plot_type, type(data), list(data.keys()) if isinstance(data, dict) else 'not_dict')
                if not data and isinstance(doc.get('text'), str):
                    # Try to parse data from the text payload: "Data: {...}."
                    text = doc['text']
                    marker = "Data: "
                    if marker in text:
                        try:
                            data_str = text.split(marker, 1)[1]
                            if "Metadata:" in data_str:
                                data_str = data_str.split("Metadata:", 1)[0]
                            data_str = data_str.strip().rstrip(". ")
                            data = json.loads(data_str)
                            logger.debug("extract: Parsed data from text, keys=%s",
                                         list(data.keys()) if isinstance(data, dict) else 'not_dict')
                        except Exception as e:
                            logger.debug("extract: Failed to parse data from text: %s", e)
                            data = None
                summary_data = data or {}
                logger.debug("extract: Final summary_data keys=%s, has_pos=%s, has_neg=%s",
                             list(summary_data.keys()) if isinstance(summary_data, dict) else 'not_dict',
                             bool(summary_data.get('positive_words') if isinstance(summary_data, dict) else False),
                             bool(summary_data.get('negative_words') if isinstance(summary_data, dict) else False))
                summaries.append({
                    'title': meta.get('title', ''),
                    'plot_type': plot_type,
                    'data': summary_data,
                    'description': meta.get('description', ''),
                    'summary_text': meta.get('summary_text', '')
                })
            return summaries
        
        plot_summaries = extract_plot_summary_data(relevant_docs)
        
        # Also search all documents for plot summaries if question is about plots/visualizations
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in ["plot", "chart", "visualization", "graph", "word", "sentiment", "positive", "negative", "image", "class", "insight", "pattern", "analyze", "extract"]):
            if user_id in vector_db.collections:
                all_docs = []
                for text, meta in zip(vector_db.collections[user_id]['documents'], 
                                    vector_db.collections[user_id]['metadata']):
                    all_docs.append({'text': text, 'metadata': meta})
                all_plot_summaries = extract_plot_summary_data(all_docs)
                # Merge unique plot summaries (avoid duplicates)
                existing_keys = {f"{s.get('plot_type')}_{s.get('title')}" for s in plot_summaries}
                for s in all_plot_summaries:
                    unique_key = f"{s.get('plot_type')}_{s.get('title')}"
                    if unique_key not in existing_keys:
                        plot_summaries.append(s)
                        existing_keys.add(unique_key)
        
        # Format context using the new formatting function
        formatted_context = format_context_for_llm(relevant_docs, plot_summaries)
        
        # Get conversation history
        conversation_context = get_conversation_context(user_id)
        
        # Extract XAI artifacts for constrained prompt
        xai_artifacts = extract_xai_artifacts_from_docs(relevant_docs)
        
        logger.debug("Context length: %d characters", len(formatted_context))
        logger.debug("Context preview: %s...", formatted_context[:200])
        logger.debug("Conversation context length: %d characters", len(conversation_context))
        logger.debug("Found %d plot summaries", len(plot_summaries))
        logger.debug("XAI artifacts - LIME: %d, Attention: %d, Confidence: %s",
                    len(xai_artifacts.get('lime_features', [])),
                    len(xai_artifacts.get('attention_tokens', [])),
                    xai_artifacts.get('confidence_score'))
        
        # Use constrained prompt if requested and artifacts available
        if use_constrained and (xai_artifacts.get('lime_features') or xai_artifacts.get('attention_tokens') or xai_artifacts.get('confidence_score')):
            system_prompt, user_prompt = generate_constrained_prompt(
                question, formatted_context, xai_artifacts, conversation_context
            )
        else:
            # Fallback to original prompt if no artifacts or not using constrained
            # Create system prompt
            system_prompt = """You are an expert data scientist and AI analyst specializing in extracting actionable insights from data analysis and visualizations. Your role is to help users understand their data deeply and make data-driven decisions.

CRITICAL: You must ALWAYS provide insights, reasoning, and actionable recommendations - never just list data.

CORE PRINCIPLES:
1. **INSIGHT EXTRACTION**: Always analyze patterns, trends, and anomalies in the data. Don't just report numbers - explain what they mean.
2. **REASONING**: Connect the dots between different data points. Explain WHY patterns exist and what they indicate.
3. **ACTIONABLE RECOMMENDATIONS**: Provide specific, actionable advice based on the insights you extract.
4. **CONTEXT-AWARE**: Understand what the user is trying to accomplish and tailor your insights accordingly.

RESPONSE STRUCTURE FOR DIFFERENT QUESTION TYPES:

**Open-ended questions** ("What insights?", "Tell me about my data", "Summarize"):
- Start with a high-level overview of the dataset
- Extract 3-5 key insights with reasoning
- Identify patterns, trends, or anomalies
- Provide actionable recommendations
- Use specific numbers and metrics from the data

**Data cleaning questions** ("How should I clean the data?", "What data quality issues?"):
- Analyze the dataset structure and identify potential issues
- Check for missing values, outliers, inconsistencies
- Recommend specific cleaning steps based on the data type
- Explain why each cleaning step is important

**Pattern/Visualization questions** ("What patterns?", "Key insights from visualizations"):
- Analyze the visualizations to identify trends
- Compare different aspects of the data
- Explain what the patterns mean in business/domain context
- Highlight surprising or important findings

**Specific data questions** (word sentiment, keywords, etc.):
- Provide the requested data
- Add context: what does this data tell us?
- Identify interesting patterns or outliers
- Suggest what actions could be taken based on this information

DATA INTERPRETATION GUIDELINES:
- **Sentiment Analysis**: Don't just list words - explain what sentiment patterns reveal about the content, audience, or domain
- **Keywords**: Identify themes and topics, explain what they indicate about the dataset focus
- **Class Distributions**: Analyze balance/imbalance, identify potential issues, suggest solutions
- **Trends**: Identify direction (increasing/decreasing), rate of change, potential causes

TONE:
- Be conversational but authoritative
- Use clear, concise language
- IMPORTANT: Do NOT use markdown formatting (no **, ##, ```, -, * bullets, etc.). Write in plain text with natural sentence structure. Use numbered lists (1. 2. 3.) only when listing steps. Separate sections with blank lines instead of headers.
- Always end with actionable next steps or recommendations

CONTEXT STRUCTURE:
- Plot summaries contain structured data - parse and analyze it deeply
- Data statistics include shape, columns, types - use these to understand data quality
- Word sentiment includes positive_words and negative_words - analyze sentiment distribution and themes
- Keywords show frequency - identify main topics and themes
- Image analysis includes class distributions - assess balance and identify issues

IMPORTANT - USING PLOT DATA:
- When plot summaries are provided, reference specific plot titles and types
- Use actual numbers, frequencies, and scores from the plot data
- Compare data across different plots to identify patterns
- Reference specific visualizations: "Based on the Word Sentiment Associations plot..." or "The Top Keywords visualization shows..."
- Don't just summarize - analyze what the plots reveal about the data

Remember: Your goal is to help users make better decisions through data insights, not just to report what's in the data."""
            
            # Create user prompt with formatted context and conversation history
            user_prompt = f"""User Question: {question}

AVAILABLE DATA AND CONTEXT:
{formatted_context}

PREVIOUS CONVERSATION:
{conversation_context if conversation_context else "None"}

INSTRUCTIONS:
1. Analyze the data deeply - don't just list facts, extract insights
2. Reference specific plots and visualizations by name when available
3. Use actual numbers, frequencies, and scores from the plot data provided
4. Identify patterns, trends, anomalies, and their implications
5. Provide reasoning: explain WHY patterns exist and what they mean
6. Give actionable recommendations based on your analysis
7. Compare data across different visualizations to find connections
8. If asked about data cleaning, analyze the dataset structure and identify quality issues
9. If asked for insights, provide 3-5 key findings with explanations
10. Be specific, analytical, and helpful - the user wants to understand their data deeply

PLOT DATA USAGE:
- If plot summaries are provided, mention specific plot titles and types
- Reference actual data from plots: "The Word Sentiment Associations plot shows that 'profit' appears X times with a sentiment score of Y"
- Compare insights across different plots to provide comprehensive analysis
- Use the structured data (positive_words, negative_words, keywords) to provide specific examples

Remember: Always provide insights and reasoning, never just report data. Reference specific plots and use actual numbers from the data provided."""
        
        logger.debug("Using OpenAI: %s", openai_client is not None)
        
        # Generate response using OpenAI or fallback
        if openai_client is None:
            # Enhanced fallback response - use formatted context and plot summaries
            return generate_structured_fallback_response(question, relevant_docs, plot_summaries, conversation_context, user_id)
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error with OpenAI API: %s", e)
            # Fallback to structured response
            return generate_structured_fallback_response(question, relevant_docs, plot_summaries, conversation_context, user_id)
        
    except Exception as e:
        logger.error("Error generating RAG response: %s", e)
        return f"I encountered an error while processing your question: {str(e)}"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'ai_outputs_rag'})


# --------------------------------------------------------------------------
# RestFS plot image endpoints
# --------------------------------------------------------------------------

@app.route('/api/store-plot-image', methods=['POST'])
def store_plot_image_endpoint():
    """Store a plot image (base64 PNG) to RestFS and register metadata."""
    try:
        data = request.json or {}
        user_id = data.get('user_id')
        plot_id = data.get('plot_id') or uuid.uuid4().hex
        image_b64 = data.get('image')
        plot_type = data.get('plot_type', 'unknown')
        title = data.get('title', '')
        description = data.get('description', '')
        summary_data = data.get('data', {})
        summary_text = data.get('summary_text', '')
        dataset_id = data.get('dataset_id', '')

        if not user_id or not image_b64:
            return jsonify({'error': 'Missing user_id or image'}), 400

        ref = save_plot_image(image_b64, user_id, plot_id,
                              metadata={'plot_type': plot_type, 'title': title})

        meta = build_plot_metadata(
            user_id=user_id, plot_type=plot_type, title=title,
            description=description, summary_text=summary_text,
            plot_summary={'title': title, 'description': description,
                         'data': summary_data, 'summary_text': summary_text},
            dataset_id=dataset_id, data_mode='text', plot_id=plot_id,
            image_ref=ref,
        )
        meta['image_ref'] = ref  # backward compat for registry readers

        plots = load_registry(user_id)
        plots.append(meta)
        save_registry(user_id, plots)

        index_plot(vector_db, user_id, {
            'title': title, 'plot_type': plot_type, 'description': description,
            'summary_text': summary_text, 'data': summary_data,
        }, source='restfs_plot_image', plot_id=plot_id, image_ref=ref,
           dataset_id=dataset_id)

        return jsonify({'message': 'Plot image stored', 'plot_id': plot_id, 'image_ref': ref})
    except Exception as e:
        logger.exception('store_plot_image_endpoint failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/plots/<plot_id>/image', methods=['GET'])
def serve_plot_image(plot_id):
    """Serve a plot PNG image by plot_id."""
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400

        plots = load_registry(user_id)
        ref = ''
        for p in plots:
            if p.get('plot_id') == plot_id:
                ref = p.get('image_ref', '')
                break

        img_bytes = load_plot_image(ref, user_id, plot_id)
        if img_bytes:
            return img_bytes, 200, {'Content-Type': 'image/png',
                                    'Cache-Control': 'public, max-age=3600'}
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        logger.exception('serve_plot_image failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/user-datasets/<user_id>', methods=['GET'])
def list_user_datasets_restfs(user_id):
    """List datasets available in RestFS + local for a user."""
    try:
        datasets = []
        try:
            import restfs_client as _restfs
            if _restfs.is_available():
                for obj in _restfs.list_user_datasets(user_id):
                    fname = obj['key'].split('/')[-1] if '/' in obj['key'] else obj['key']
                    datasets.append({'filename': fname, 'key': obj['key'],
                                     'size': obj.get('size', 0),
                                     'last_modified': obj.get('last_modified'),
                                     'source': 'restfs'})
        except Exception:
            pass

        uploads_dir = os.path.join(SHARED_DATA_DIR, 'uploads')
        if os.path.isdir(uploads_dir):
            for fname in os.listdir(uploads_dir):
                fpath = os.path.join(uploads_dir, fname)
                if os.path.isfile(fpath) and fname.endswith(('.csv', '.json', '.txt')):
                    datasets.append({'filename': fname, 'key': fpath,
                                     'size': os.path.getsize(fpath),
                                     'last_modified': datetime.fromtimestamp(
                                         os.path.getmtime(fpath)).isoformat(),
                                     'source': 'local'})

        seen = set()
        unique = []
        for d in datasets:
            if d['filename'] not in seen:
                seen.add(d['filename'])
                unique.append(d)
        return jsonify({'datasets': unique})
    except Exception as e:
        logger.exception('list_user_datasets_restfs failed')
        return jsonify({'error': str(e)}), 500

@app.route('/store-data', methods=['POST'])
def store_data():
    """Store user data information"""
    try:
        data = request.json
        user_id = data.get('user_id')
        data_info = data.get('data_info', {})
        
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        
        # Store data info in vector database
        data_doc = f"Dataset: Shape {data_info.get('shape', 'Unknown')}. "
        if 'columns' in data_info:
            columns = ', '.join(data_info['columns'])
            data_doc += f"Columns: {columns}. "
        if 'data_type' in data_info:
            data_doc += f"Data type: {data_info['data_type']}."
        
        vector_db.add_document(
            user_id=user_id,
            text=data_doc,
            metadata={
                'doc_type': 'data_info',
                'timestamp': datetime.now().isoformat(),
                'data_info': data_info
            }
        )
        
        return jsonify({'message': 'Data information stored successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/store-interactive-plot', methods=['POST'])
def store_interactive_plot():
    """Store interactive plot metadata for chat grounding and plot registry"""
    try:
        data = request.json or {}
        user_id = data.get('user_id')
        plot_data = data.get('plot_data', {})
        query = data.get('query', '')

        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400

        plot_type = plot_data.get('plot_type', 'interactive')
        metadata = plot_data.get('metadata', {})
        plot_spec = metadata.get('plot_spec', {})
        plot_summary = metadata.get('plot_summary', {})
        plot_html = plot_data.get('plot_html', '')
        dataset_id = data.get('dataset_id', '')
        file_path = data.get('file_path', '')
        file_name = data.get('file_name', '')
        data_mode = data.get('data_mode', metadata.get('data_mode', 'image'))
        tags = data.get('tags') or []

        # Build full metadata for registry (new schema)
        meta = build_plot_metadata(
            user_id=user_id,
            plot_type=plot_type,
            title=(plot_summary or {}).get('title', f'Interactive Plot: {query[:50]}'),
            description=(plot_summary or {}).get('description', ''),
            query=query,
            plot_spec=plot_spec,
            plot_summary=plot_summary,
            dataset_id=dataset_id,
            file_path=file_path,
            file_name=file_name,
            data_mode=data_mode,
            tags=tags,
        )

        # Save plot HTML to disk and set path in metadata
        if plot_html:
            html_path = save_plot_html(plot_html, user_id, meta['plot_id'])
            meta['storage']['html_ref'] = html_path
            meta['plot_html_path'] = html_path  # backward compat

        # Append to user's plot registry
        plots = load_registry(user_id)
        plots.append(meta)
        save_registry(user_id, plots)

        # RAG: Store as interactive_plot document
        doc = f"Interactive Plot Request: {query}. Plot type: {plot_type}. "
        if plot_spec:
            doc += f"Plot spec: {json.dumps(plot_spec, default=str)}. "
        if plot_summary:
            summary_data = plot_summary.get('data', {})
            if summary_data:
                doc += f"Data: {json.dumps(summary_data, default=str)}. "
            if plot_summary.get('summary_text'):
                doc += f"Summary: {plot_summary.get('summary_text')}. "
        if metadata:
            doc += f"Metadata: {json.dumps(metadata, default=str)}."

        vector_db.add_document(
            user_id,
            doc,
            {
                'doc_type': 'interactive_plot',
                'plot_type': plot_type,
                'query': query,
                'plot_spec': plot_spec,
                'plot_summary': plot_summary,
                'plot_id': meta['plot_id'],
                'timestamp': datetime.now().isoformat()
            }
        )

        if plot_summary and isinstance(plot_summary, dict):
            title = plot_summary.get('title', f'Interactive Plot: {query}')
            plot_type_from_summary = plot_summary.get('plot_type', plot_type)
            description = plot_summary.get('description', '')
            summary_data = plot_summary.get('data', {})
            summary_text = plot_summary.get('summary_text', '')
            summary_doc = f"{title}. Plot type: {plot_type_from_summary}. "
            if description:
                summary_doc += f"Description: {description}. "
            if summary_text:
                summary_doc += f"Summary: {summary_text}. "
            if summary_data:
                summary_doc += f"Data: {json.dumps(summary_data, default=str)}. "
            vector_db.add_document(
                user_id,
                summary_doc,
                {
                    'doc_type': 'plot_summary',
                    'plot_type': plot_type_from_summary,
                    'source': 'interactive_plot',
                    'title': title,
                    'description': description,
                    'data': summary_data,
                    'summary_text': summary_text,
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }
            )

        return jsonify({
            'message': 'Interactive plot stored successfully',
            'plot_id': meta['plot_id']
        })
    except Exception as e:
        logger.exception('store_interactive_plot failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/plots', methods=['GET'])
def list_plots():
    """List saved plots for a user with optional filters."""
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        dataset_id = request.args.get('dataset_id', '')
        plot_type = request.args.get('plot_type', '')
        limit = min(int(request.args.get('limit', 100)), 500)
        offset = max(0, int(request.args.get('offset', 0)))

        plots = load_registry(user_id)
        if dataset_id:
            plots = [p for p in plots if p.get('dataset_id') == dataset_id]
        if plot_type:
            plots = [p for p in plots if p.get('plot_type') == plot_type]
        total = len(plots)
        plots = plots[offset:offset + limit]
        return jsonify({'plots': plots, 'total': total, 'limit': limit, 'offset': offset})
    except Exception as e:
        logger.exception('list_plots failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/plots/<plot_id>', methods=['GET'])
def get_plot(plot_id):
    """Get metadata for a single plot."""
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        plots = load_registry(user_id)
        for p in plots:
            if p.get('plot_id') == plot_id:
                return jsonify(p)
        return jsonify({'error': 'Plot not found'}), 404
    except Exception as e:
        logger.exception('get_plot failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/plots/<plot_id>/html', methods=['GET'])
def get_plot_html(plot_id):
    """Get plot HTML content."""
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        plots = load_registry(user_id)
        for p in plots:
            if p.get('plot_id') == plot_id:
                path = p.get('plot_html_path', '')
                loc = p.get('storage_location', '')
                html = load_plot_html(loc, path)
                if html is not None:
                    return html, 200, {'Content-Type': 'text/html; charset=utf-8'}
                return jsonify({'error': 'Plot HTML not available'}), 404
        return jsonify({'error': 'Plot not found'}), 404
    except Exception as e:
        logger.exception('get_plot_html failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/plots/<plot_id>', methods=['DELETE'])
def delete_plot(plot_id):
    """Delete a saved plot."""
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        plots = load_registry(user_id)
        for i, p in enumerate(plots):
            if p.get('plot_id') == plot_id:
                delete_plot_file(p.get('plot_html_path', ''))
                plots.pop(i)
                save_registry(user_id, plots)
                return jsonify({'message': 'Plot deleted'})
        return jsonify({'error': 'Plot not found'}), 404
    except Exception as e:
        logger.exception('delete_plot failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/<user_id>', methods=['GET'])
def list_datasets(user_id):
    """List datasets (unique file_name/dataset_id) for a user with plot counts."""
    try:
        plots = load_registry(user_id)
        seen = {}
        for p in plots:
            did = p.get('dataset_id') or p.get('file_name') or 'unknown'
            name = p.get('file_name') or did
            if did not in seen:
                seen[did] = {'dataset_id': did, 'file_name': name, 'plot_count': 0}
            seen[did]['plot_count'] += 1
        return jsonify({'datasets': list(seen.values())})
    except Exception as e:
        logger.exception('list_datasets failed')
        return jsonify({'error': str(e)}), 500


@app.route('/store-results', methods=['POST'])
def store_results():
    """Store XAI results in vector database"""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        
        # Store results in vector database
        store_results_in_vector_db(data, user_id)
        
        return jsonify({'message': 'Results stored successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/store-attention-insights', methods=['POST'])
def store_attention_insights():
    """Store attention analysis insights for AI assistant access"""
    try:
        data = request.json
        user_id = data.get('user_id')
        attention_analysis = data.get('attention_analysis', {})
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Create attention insights document for vector database
        example_text = attention_analysis.get('example_text', '')
        insights = attention_analysis.get('insights', {})
        
        # Generate detailed attention insights document
        attention_doc = f"Attention Analysis for: '{example_text[:100]}...'\n\n"
        
        # Top tokens information
        top_tokens = insights.get('top_tokens', [])
        if top_tokens:
            attention_doc += "Top Important Tokens:\n"
            for i, (token, score) in enumerate(top_tokens[:5], 1):
                attention_doc += f"{i}. '{token}' (score: {score:.3f})\n"
            attention_doc += "\n"
        
        # Attention metrics
        max_score = insights.get('max_attention_score', 0)
        variance = insights.get('attention_variance', 0)
        concentration = insights.get('attention_concentration', 'unknown')
        sentiment_corr = insights.get('sentiment_correlation', 'neutral')
        
        attention_doc += f"Attention Metrics:\n"
        attention_doc += f"- Maximum attention score: {max_score:.3f}\n"
        attention_doc += f"- Attention variance: {variance:.3f}\n"
        attention_doc += f"- Attention pattern: {concentration}\n"
        attention_doc += f"- Sentiment correlation: {sentiment_corr}\n"
        
        # Store in vector database
        vector_db.add_document(
            user_id,
            attention_doc,
            {
            'doc_type': 'attention_analysis',
                'example_index': attention_analysis.get('example_index', ''),
                'model_type': attention_analysis.get('model_type', ''),
            'insights': insights,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        return jsonify({'message': 'Attention insights stored successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<user_id>', methods=['GET'])
def get_results(user_id):
    """Get summary of stored results for a user"""
    try:
        if user_id not in vector_db.collections:
            return jsonify({'error': 'No results found for user'}), 404
        
        collection = vector_db.collections[user_id]
        
        # Safely extract document types and timestamps
        document_types = []
        timestamps = []
        
        for doc in collection['metadata']:
            if 'doc_type' in doc:
                document_types.append(doc['doc_type'])
            if 'timestamp' in doc:
                timestamps.append(doc['timestamp'])
        
        summary = {
            'user_id': user_id,
            'total_documents': len(collection['documents']),
            'document_types': list(set(document_types)),
            'last_updated': max(timestamps) if timestamps else None
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat questions using RAG approach"""
    try:
        data = request.json
        question = data.get('question')
        user_id = data.get('user_id')
        
        if not question or not user_id:
            return jsonify({'error': 'Missing question or user_id'}), 400
        
        # Rehydrate from RustFS if vector DB is empty for this user
        if not vector_db.has_documents(user_id):
            rehydrate_from_restfs(vector_db, user_id)
        
        # Check if user has any stored results
        if user_id not in vector_db.collections:
            return jsonify({'error': 'No results found for user. Please upload and analyze a model first.'}), 404
        
        # Generate RAG response
        answer = generate_rag_response(question, user_id)
        
        # Store conversation history
        add_to_conversation_history(user_id, question, answer)
        
        return jsonify({
            'answer': answer,
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-user-data/<user_id>', methods=['DELETE'])
def clear_user_data(user_id):
    """Clear all data for a specific user"""
    try:
        if user_id in vector_db.collections:
            del vector_db.collections[user_id]
        if user_id in conversation_history:
            del conversation_history[user_id]
        return jsonify({'message': f'All data cleared for user {user_id}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/evaluate-faithfulness', methods=['POST'])
def evaluate_faithfulness():
    """
    Evaluate faithfulness of RAG responses on test set
    
    Request body:
    {
        "user_id": "admin",
        "test_set_path": "ai_outputs/test_set.json" (optional)
    }
    
    Returns:
    {
        "constrained_prompt": {
            "grounding_percentage": 0.85,
            "hallucination_rate": 0.08,
            "avg_citations_per_response": 2.3,
            "feature_overlap": 0.78
        },
        "naive_prompt": {
            "grounding_percentage": 0.45,
            "hallucination_rate": 0.35,
            "avg_citations_per_response": 0.2,
            "feature_overlap": 0.42
        },
        "improvement": {
            "grounding_delta": 0.40,
            "hallucination_reduction": 0.27
        }
    }
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        test_set_path = data.get('test_set_path', 'ai_outputs/test_set.json')
        
        if not user_id:
            return jsonify({'error': 'Missing user_id'}), 400
        
        # Check if user has stored results
        if user_id not in vector_db.collections:
            return jsonify({'error': 'No results found for user. Please upload and analyze a model first.'}), 404
        
        # Initialize evaluator and test set generator
        evaluator = FaithfulnessEvaluator()
        test_set_gen = TestSetGenerator()
        
        # Load or create test set
        if os.path.exists(test_set_path):
            test_cases = test_set_gen.load_test_set(test_set_path)
        else:
            # Try to create test set from user's stored XAI artifacts
            # Search for XAI analysis documents
            all_docs = []
            if user_id in vector_db.collections:
                for text, meta in zip(vector_db.collections[user_id]['documents'], 
                                    vector_db.collections[user_id]['metadata']):
                    all_docs.append({'text': text, 'metadata': meta})
            
            # Extract XAI artifacts from stored documents
            xai_docs = [doc for doc in all_docs if doc.get('metadata', {}).get('doc_type') == 'xai_analysis']
            
            if not xai_docs:
                return jsonify({'error': 'No XAI analysis results found. Please run XAI analysis first.'}), 404
            
            # Get the most recent XAI analysis
            latest_xai = xai_docs[-1]
            metadata = latest_xai.get('metadata', {})
            
            lime_features = metadata.get('lime_features', [])
            attention_tokens = metadata.get('attention_tokens', [])
            confidence_score = metadata.get('confidence_score')
            example_text = metadata.get('example_text', '')
            
            if not lime_features and not attention_tokens:
                return jsonify({'error': 'XAI artifacts not found in stored results. Please run XAI analysis first.'}), 404
            
            # Create test set from artifacts
            test_cases = test_set_gen.create_test_set_from_artifacts(
                lime_features, attention_tokens, confidence_score or 0.0, example_text
            )
            
            # Save test set for future use
            os.makedirs(os.path.dirname(test_set_path) if os.path.dirname(test_set_path) else '.', exist_ok=True)
            test_set_gen.save_test_set(test_cases, test_set_path)
        
        if not test_cases:
            return jsonify({'error': 'Test set is empty'}), 404
        
        # Run evaluation
        constrained_results = []
        naive_results = []
        
        for test_case in test_cases:
            prompt = test_case['prompt']
            
            # Extract expected artifacts
            expected_artifacts = {
                'expected_features': test_case.get('expected_features', []),
                'expected_tokens': test_case.get('expected_tokens', []),
                'expected_values': test_case.get('expected_values', {})
            }
            
            # Generate constrained response
            constrained_response = generate_rag_response(prompt, user_id, use_constrained=True)
            constrained_eval = evaluator.evaluate_response(constrained_response, expected_artifacts)
            constrained_results.append(constrained_eval)
            
            # Generate naive response
            naive_response = generate_naive_rag_response(prompt, user_id)
            naive_eval = evaluator.evaluate_response(naive_response, expected_artifacts)
            naive_results.append(naive_eval)
        
        # Aggregate metrics
        def aggregate_metrics(eval_results):
            total = len(eval_results)
            if total == 0:
                return {}
            
            grounded_count = sum(1 for r in eval_results if r.get('overall_grounded', False))
            hallucination_count = sum(1 for r in eval_results if r.get('hallucinations', {}).get('has_hallucinations', False))
            total_citations = sum(r.get('citations', {}).get('citation_count', 0) for r in eval_results)
            
            # Feature overlap (average)
            feature_overlaps = [r.get('feature_overlap', 0.0) for r in eval_results if r.get('feature_overlap', 0.0) > 0]
            avg_feature_overlap = sum(feature_overlaps) / len(feature_overlaps) if feature_overlaps else 0.0
            
            return {
                'grounding_percentage': grounded_count / total,
                'hallucination_rate': hallucination_count / total,
                'avg_citations_per_response': total_citations / total,
                'feature_overlap': avg_feature_overlap,
                'total_responses': total
            }
        
        constrained_metrics = aggregate_metrics(constrained_results)
        naive_metrics = aggregate_metrics(naive_results)
        
        # Calculate improvement
        improvement = {
            'grounding_delta': constrained_metrics.get('grounding_percentage', 0.0) - naive_metrics.get('grounding_percentage', 0.0),
            'hallucination_reduction': naive_metrics.get('hallucination_rate', 0.0) - constrained_metrics.get('hallucination_rate', 0.0),
            'citation_improvement': constrained_metrics.get('avg_citations_per_response', 0.0) - naive_metrics.get('avg_citations_per_response', 0.0),
            'feature_overlap_improvement': constrained_metrics.get('feature_overlap', 0.0) - naive_metrics.get('feature_overlap', 0.0)
        }
        
        return jsonify({
            'constrained_prompt': constrained_metrics,
            'naive_prompt': naive_metrics,
            'improvement': improvement,
            'test_set_size': len(test_cases),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True) 
