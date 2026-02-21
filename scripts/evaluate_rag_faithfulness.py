#!/usr/bin/env python3
"""
Standalone script to evaluate RAG faithfulness

Usage:
    python scripts/evaluate_rag_faithfulness.py --user_id admin --test_set ai_outputs/test_set.json
"""

import argparse
import json
import sys
import os
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    sys.exit(1)


def run_evaluation(user_id: str, test_set_path: str = None, base_url: str = "http://localhost:8001"):
    """
    Run faithfulness evaluation via API endpoint
    
    Args:
        user_id: User identifier
        test_set_path: Path to test set JSON file (optional)
        base_url: Base URL for API (default: http://localhost:8001)
    """
    print(f"Running RAG faithfulness evaluation for user: {user_id}")
    print(f"API endpoint: {base_url}")
    print("-" * 60)
    
    # Prepare request
    payload = {
        'user_id': user_id
    }
    
    if test_set_path:
        payload['test_set_path'] = test_set_path
    
    try:
        # Make API request
        response = requests.post(
            f"{base_url}/evaluate-faithfulness",
            json=payload,
            timeout=300  # 5 minute timeout for evaluation
        )
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        results = response.json()
        
        # Display results
        print("\n" + "=" * 60)
        print("FAITHFULNESS EVALUATION RESULTS")
        print("=" * 60)
        
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
        
        # Save results to file
        output_file = f"faithfulness_evaluation_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
        
        return results
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {base_url}")
        print("Make sure the AI outputs service is running.")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The evaluation may take a while.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Evaluate RAG faithfulness for XAI explanations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default test set
  python scripts/evaluate_rag_faithfulness.py --user_id admin
  
  # Evaluate with custom test set
  python scripts/evaluate_rag_faithfulness.py --user_id admin --test_set ai_outputs/test_set.json
  
  # Use custom API URL
  python scripts/evaluate_rag_faithfulness.py --user_id admin --base_url http://localhost:8001
        """
    )
    
    parser.add_argument(
        '--user_id',
        type=str,
        required=True,
        help='User identifier'
    )
    
    parser.add_argument(
        '--test_set',
        type=str,
        default=None,
        help='Path to test set JSON file (optional, will be created from XAI artifacts if not provided)'
    )
    
    parser.add_argument(
        '--base_url',
        type=str,
        default='http://localhost:8001',
        help='Base URL for AI outputs API (default: http://localhost:8001)'
    )
    
    args = parser.parse_args()
    
    results = run_evaluation(
        user_id=args.user_id,
        test_set_path=args.test_set,
        base_url=args.base_url
    )
    
    if results:
        print("\n" + "=" * 60)
        print("Evaluation completed successfully!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("Evaluation failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()
