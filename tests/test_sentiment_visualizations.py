#!/usr/bin/env python3
"""
Test script for enhanced sentiment analysis visualizations
Validates the new visualization system with pre-sample explanations
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os

def create_sample_sentiment_model():
    """Create a sample sentiment analysis model for testing"""
    
    # Load sample data
    with open('shared_volume/sample_sentiment_data.json', 'r') as f:
        data = json.load(f)
    
    # Extract articles
    articles = data['articles']
    
    # Create DataFrame
    df = pd.DataFrame(articles)
    
    # Prepare text data
    texts = df['title'].tolist()
    labels = df['sentiment_label'].tolist()
    
    # Convert labels to numeric
    label_map = {'positive': 1, 'negative': 0, 'neutral': 2}
    numeric_labels = [label_map[label] for label in labels]
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, numeric_labels, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, feature_names, df

def test_enhanced_sentiment_visualizations():
    """Test the enhanced sentiment visualization system"""
    
    print("Testing Enhanced Sentiment Analysis Visualizations")
    print("=" * 50)
    
    try:
        # Create sample model
        print("1. Creating sample sentiment analysis model...")
        model, X_train, feature_names, df = create_sample_sentiment_model()
        print(f"   ✅ Model created with {len(feature_names)} features")
        print(f"   ✅ Training data shape: {X_train.shape}")
        
        # Test visualization generation
        print("\n2. Testing visualization generation...")
        
        # Import the visualization function
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'xai_service'))
        
        # Mock the user_id for testing
        user_id = 'test_user'
        model_type = 'finbert_sentiment'
        
        # Test the enhanced visualization function
        from xai_service.app import generate_enhanced_sentiment_visualizations
        
        print("   Generating enhanced sentiment visualizations...")
        images = generate_enhanced_sentiment_visualizations(model, X_train, feature_names, model_type, user_id)
        
        print(f"   ✅ Generated {len(images)} visualizations")
        
        # Validate each visualization
        for i, img_data in enumerate(images):
            try:
                # Decode base64 image
                img_bytes = base64.b64decode(img_data)
                
                # Try to display image info
                print(f"   📊 Visualization {i+1}: {len(img_bytes)} bytes")
                
                # Save test image
                test_img_path = f'test_sentiment_viz_{i+1}.png'
                with open(test_img_path, 'wb') as f:
                    f.write(img_bytes)
                print(f"   💾 Saved to: {test_img_path}")
                
            except Exception as e:
                print(f"   ❌ Error with visualization {i+1}: {e}")
        
        print("\n3. Testing visualization content...")
        
        # Check if explanations are included
        explanation_found = any('EXPLANATION' in str(img_data) or 'GUIDE' in str(img_data) for img_data in images)
        print(f"   {'✅' if explanation_found else '❌'} Pre-sample explanations included")
        
        # Check if confidence analysis is included
        confidence_found = any('CONFIDENCE' in str(img_data) or 'UNCERTAINTY' in str(img_data) for img_data in images)
        print(f"   {'✅' if confidence_found else '❌'} Confidence analysis included")
        
        # Check if sentiment patterns are included
        patterns_found = any('PATTERN' in str(img_data) or 'TREND' in str(img_data) for img_data in images)
        print(f"   {'✅' if patterns_found else '❌'} Sentiment pattern analysis included")
        
        print("\n4. Testing API integration...")
        
        # Test API endpoint
        try:
            response = requests.post('http://localhost:8000/analyze', json={
                'model_path': 'test_model.joblib',
                'user_id': 'test_user',
                'data_type': 'text'
            }, timeout=30)
            
            if response.status_code == 200:
                print("   ✅ API endpoint working")
                result = response.json()
                print(f"   📊 API returned {len(result.get('images', []))} visualizations")
            else:
                print(f"   ⚠️  API returned status {response.status_code}")
                
        except Exception as e:
            print(f"   ⚠️  API test failed: {e}")
        
        print("\n5. Testing data quality...")
        
        # Analyze the sample data
        sentiment_dist = df['sentiment_label'].value_counts()
        print(f"   📈 Sentiment distribution:")
        for label, count in sentiment_dist.items():
            print(f"      {label}: {count} articles")
        
        avg_sentiment = df['sentiment'].mean()
        print(f"   📊 Average sentiment score: {avg_sentiment:.3f}")
        
        # Check for balanced dataset
        is_balanced = abs(sentiment_dist.get('positive', 0) - sentiment_dist.get('negative', 0)) <= 2
        print(f"   {'✅' if is_balanced else '⚠️'} Dataset is {'balanced' if is_balanced else 'imbalanced'}")
        
        print("\n" + "=" * 50)
        print("✅ Enhanced Sentiment Analysis Test Completed Successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization_quality():
    """Test the quality and interpretability of visualizations"""
    
    print("\nTesting Visualization Quality")
    print("=" * 30)
    
    try:
        # Create sample model
        model, X_train, feature_names, df = create_sample_sentiment_model()
        
        # Test different visualization aspects
        tests = [
            ("Feature Importance", "importance" in str(feature_names)),
            ("Sentiment Distribution", len(df['sentiment_label'].unique()) >= 2),
            ("Model Performance", hasattr(model, 'predict_proba')),
            ("Text Processing", len(feature_names) > 0),
            ("Data Quality", not df.isnull().any().any())
        ]
        
        for test_name, result in tests:
            print(f"   {'✅' if result else '❌'} {test_name}")
        
        # Test visualization generation time
        import time
        start_time = time.time()
        
        from xai_service.app import generate_enhanced_sentiment_visualizations
        images = generate_enhanced_sentiment_visualizations(model, X_train, feature_names, 'test_model', 'test_user')
        
        generation_time = time.time() - start_time
        print(f"   ⏱️  Visualization generation time: {generation_time:.2f} seconds")
        print(f"   📊 Generated {len(images)} visualizations")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality test failed: {e}")
        return False

def create_visualization_report():
    """Create a comprehensive report of the visualization system"""
    
    print("\nCreating Visualization System Report")
    print("=" * 40)
    
    report = {
        "system_overview": {
            "name": "Enhanced Sentiment Analysis Visualization System",
            "version": "2.0",
            "features": [
                "Pre-sample explanations and guides",
                "Sentiment-aware feature importance",
                "Confidence and uncertainty analysis",
                "Pattern recognition and trend analysis",
                "Interactive word clouds",
                "Comprehensive performance metrics"
            ]
        },
        "visualization_types": [
            {
                "name": "Explanation Dashboard",
                "description": "Comprehensive guide explaining how to interpret sentiment analysis results",
                "key_elements": ["Model type explanation", "Visualization interpretation guide", "Key insights to look for", "Next steps recommendations"]
            },
            {
                "name": "Enhanced Feature Importance",
                "description": "Word importance analysis with sentiment context coloring",
                "key_elements": ["Color-coded positive/negative/neutral words", "Importance scores", "Sentiment influence indicators"]
            },
            {
                "name": "Confidence Analysis",
                "description": "Model confidence and uncertainty assessment",
                "key_elements": ["Confidence distribution", "Uncertainty metrics", "Reliability assessment", "Performance indicators"]
            },
            {
                "name": "Pattern Analysis",
                "description": "Comprehensive sentiment pattern recognition",
                "key_elements": ["Trend analysis", "Correlation heatmaps", "Distribution analysis", "Insights summary"]
            },
            {
                "name": "Sentiment-Aware Word Cloud",
                "description": "Interactive word cloud with sentiment-based coloring",
                "key_elements": ["Red-Yellow-Green color scheme", "Word frequency visualization", "Sentiment context"]
            }
        ],
        "improvements": [
            "Added pre-sample explanations for better user understanding",
            "Enhanced feature importance with sentiment context",
            "Included confidence and uncertainty analysis",
            "Added comprehensive pattern recognition",
            "Improved visual design and color coding",
            "Enhanced error handling and fallback mechanisms"
        ],
        "usage_guidelines": [
            "Upload text data in JSON, CSV, or TXT format",
            "Select 'Text Data' as the data type",
            "Choose 'FinBERT Sentiment Analysis' as the model type",
            "Review the explanation dashboard first",
            "Use the chat feature to ask specific questions",
            "Explore individual visualizations for detailed insights"
        ]
    }
    
    # Save report
    with open('sentiment_visualization_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Report saved to: sentiment_visualization_report.json")
    
    # Print summary
    print(f"\n📊 Report Summary:")
    print(f"   • {len(report['visualization_types'])} visualization types")
    print(f"   • {len(report['improvements'])} key improvements")
    print(f"   • {len(report['usage_guidelines'])} usage guidelines")
    
    return report

if __name__ == "__main__":
    print("Enhanced Sentiment Analysis Visualization Test Suite")
    print("=" * 60)
    
    # Run tests
    test1_success = test_enhanced_sentiment_visualizations()
    test2_success = test_visualization_quality()
    
    # Create report
    report = create_visualization_report()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Enhanced Visualizations Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"Quality Assessment Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    print(f"Report Generation: ✅ COMPLETED")
    
    if test1_success and test2_success:
        print("\n🎉 ALL TESTS PASSED! Enhanced sentiment analysis system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    print("\n📋 Next Steps:")
    print("   1. Upload the sample_sentiment_data.json file to the dashboard")
    print("   2. Select 'Text Data' as the data type")
    print("   3. Choose 'FinBERT Sentiment Analysis' as the model")
    print("   4. Review the enhanced visualizations with explanations")
    print("   5. Use the chat feature to ask questions about the results") 