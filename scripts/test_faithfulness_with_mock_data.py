#!/usr/bin/env python3
"""
Test faithfulness evaluation with mock XAI artifacts

This script directly stores mock XAI artifacts in the vector DB
and then tests the faithfulness evaluation endpoint.
"""

import sys
import os
import requests
import json
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE_URL = "http://localhost:8002"


def store_mock_xai_artifacts():
    """Store mock XAI artifacts in the vector DB via API"""
    print("Storing mock XAI artifacts...")
    
    # Mock XAI analysis results with structured artifacts
    xai_results = {
        'user_id': 'admin',
        'xai_analysis': {
            'example_text': 'The company reported strong profit growth and positive revenue trends',
            'prediction': [{'label': 'positive', 'score': 0.85}],
            'visualizations': ['lime', 'attention'],
            'model_type': 'finbert',
            'analysis_type': 'sentiment_analysis',
            'timestamp': '2026-02-05T18:00:00',
            # Structured XAI artifacts for faithfulness evaluation
            'lime_features': [
                ('profit', 0.52),
                ('growth', 0.48),
                ('positive', 0.45),
                ('revenue', 0.42),
                ('strong', 0.38),
                ('company', 0.35),
                ('reported', 0.32),
                ('trends', 0.28),
                ('and', 0.25),
                ('the', 0.22)
            ],
            'attention_tokens': [
                ('profit', 0.58),
                ('growth', 0.55),
                ('positive', 0.52),
                ('revenue', 0.48),
                ('strong', 0.45),
                ('company', 0.42),
                ('reported', 0.38),
                ('trends', 0.35),
                ('and', 0.32),
                ('the', 0.28)
            ],
            'confidence_score': 0.85,
            'example_index': 0
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/store-results",
            json=xai_results,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✓ Mock XAI artifacts stored successfully")
            return True
        else:
            print(f"✗ Failed to store artifacts: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Error storing artifacts: {e}")
        return False


def test_faithfulness_evaluation():
    """Test the faithfulness evaluation endpoint"""
    print("\nRunning faithfulness evaluation...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/evaluate-faithfulness",
            json={'user_id': 'admin'},
            timeout=120  # 2 minute timeout for evaluation
        )
        
        if response.status_code != 200:
            print(f"✗ Evaluation failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
        
        results = response.json()
        
        print("\n" + "="*60)
        print("FAITHFULNESS EVALUATION RESULTS")
        print("="*60)
        
        print("\n--- CONSTRAINED PROMPT METRICS ---")
        constrained = results.get('constrained_prompt', {})
        print(f"Grounding Percentage: {constrained.get('grounding_percentage', 0.0):.2%}")
        print(f"Hallucination Rate: {constrained.get('hallucination_rate', 0.0):.2%}")
        print(f"Avg Citations per Response: {constrained.get('avg_citations_per_response', 0.0):.2f}")
        print(f"Feature Overlap: {constrained.get('feature_overlap', 0.0):.2%}")
        print(f"Total Responses: {constrained.get('total_responses', 0)}")
        
        print("\n--- NAIVE PROMPT METRICS ---")
        naive = results.get('naive_prompt', {})
        print(f"Grounding Percentage: {naive.get('grounding_percentage', 0.0):.2%}")
        print(f"Hallucination Rate: {naive.get('hallucination_rate', 0.0):.2%}")
        print(f"Avg Citations per Response: {naive.get('avg_citations_per_response', 0.0):.2f}")
        print(f"Feature Overlap: {naive.get('feature_overlap', 0.0):.2%}")
        print(f"Total Responses: {naive.get('total_responses', 0)}")
        
        print("\n--- IMPROVEMENT (Constrained vs Naive) ---")
        improvement = results.get('improvement', {})
        print(f"Grounding Delta: {improvement.get('grounding_delta', 0.0):.2%}")
        print(f"Hallucination Reduction: {improvement.get('hallucination_reduction', 0.0):.2%}")
        print(f"Citation Improvement: {improvement.get('citation_improvement', 0.0):.2f}")
        print(f"Feature Overlap Improvement: {improvement.get('feature_overlap_improvement', 0.0):.2%}")
        
        print(f"\nTest Set Size: {results.get('test_set_size', 0)}")
        print(f"Timestamp: {results.get('timestamp', 'N/A')}")
        
        # Save results
        output_file = f"faithfulness_test_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
        
        # Check if results are reasonable
        if constrained.get('grounding_percentage', 0) > 0:
            print("\n✓ Evaluation completed successfully!")
            return True
        else:
            print("\n⚠ Evaluation completed but no grounded responses found")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Evaluation timed out (may take longer than expected)")
        return False
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("="*60)
    print("FAITHFULNESS EVALUATION TEST WITH MOCK DATA")
    print("="*60)
    
    # Check if service is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"✗ Service not healthy: {response.status_code}")
            return 1
        print("✓ Service is healthy")
    except Exception as e:
        print(f"✗ Cannot connect to service at {BASE_URL}: {e}")
        print("  Make sure Docker services are running: docker-compose up")
        return 1
    
    # Store mock artifacts
    if not store_mock_xai_artifacts():
        return 1
    
    # Wait a moment for storage to complete
    time.sleep(2)
    
    # Run evaluation
    if not test_faithfulness_evaluation():
        return 1
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
