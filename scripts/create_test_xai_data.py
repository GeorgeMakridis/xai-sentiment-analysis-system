#!/usr/bin/env python3
"""
Create mock XAI analysis data for faithfulness evaluation testing
"""

import requests
import json
import sys

def create_mock_xai_data(user_id: str = "admin", base_url: str = "http://localhost:8002"):
    """
    Create and store mock XAI analysis results for testing
    """
    print(f"Creating mock XAI data for user: {user_id}")
    print("-" * 60)
    
    # Mock XAI analysis with structured artifacts
    mock_xai_analysis = {
        "user_id": user_id,
        "xai_analysis": {
            "example_text": "The company reported strong earnings growth this quarter, with revenue increasing by 25% and positive market sentiment.",
            "prediction": {
                "label": "positive",
                "score": 0.87
            },
            "model_type": "finbert",
            "analysis_type": "sentiment_analysis",
            "timestamp": "2026-02-05T20:00:00",
            "visualizations": ["lime_plot", "attention_plot", "confidence_plot"],
            # Structured XAI artifacts for faithfulness evaluation
            "lime_features": [
                ("earnings", 0.15),
                ("growth", 0.12),
                ("revenue", 0.11),
                ("positive", 0.10),
                ("strong", 0.09),
                ("increasing", 0.08),
                ("company", 0.07),
                ("quarter", 0.06),
                ("market", 0.05),
                ("sentiment", 0.04)
            ],
            "attention_tokens": [
                ("earnings", 0.18),
                ("growth", 0.16),
                ("revenue", 0.14),
                ("positive", 0.12),
                ("strong", 0.10),
                ("increasing", 0.09),
                ("25%", 0.08),
                ("company", 0.07),
                ("quarter", 0.06),
                ("market", 0.05)
            ],
            "confidence_score": 0.87
        }
    }
    
    try:
        # Store the mock data
        response = requests.post(
            f"{base_url}/store-results",
            json=mock_xai_analysis,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✓ Mock XAI data stored successfully")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Failed to store data: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Could not connect to API at {base_url}")
        print("  Make sure the AI outputs service is running.")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create mock XAI data for testing')
    parser.add_argument('--user_id', type=str, default='admin', help='User identifier')
    parser.add_argument('--base_url', type=str, default='http://localhost:8002', 
                       help='Base URL for API')
    
    args = parser.parse_args()
    
    success = create_mock_xai_data(args.user_id, args.base_url)
    sys.exit(0 if success else 1)
