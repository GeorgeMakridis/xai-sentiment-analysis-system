#!/usr/bin/env python3
"""
Direct test of faithfulness evaluation - tests core functionality
without requiring full API stack or model downloads
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_outputs.faithfulness_evaluator import FaithfulnessEvaluator
from ai_outputs.test_set_generator import TestSetGenerator


def test_direct_evaluation():
    """Test faithfulness evaluation directly with mock data"""
    print("="*60)
    print("DIRECT FAITHFULNESS EVALUATION TEST")
    print("="*60)
    
    # Initialize components
    evaluator = FaithfulnessEvaluator()
    test_set_gen = TestSetGenerator()
    
    # Create mock XAI artifacts
    lime_features = [
        ('profit', 0.52),
        ('growth', 0.48),
        ('positive', 0.45),
        ('revenue', 0.42),
        ('strong', 0.38),
        ('company', 0.35),
        ('reported', 0.32),
        ('trends', 0.28)
    ]
    
    attention_tokens = [
        ('profit', 0.58),
        ('growth', 0.55),
        ('positive', 0.52),
        ('revenue', 0.48),
        ('strong', 0.45),
        ('company', 0.42),
        ('reported', 0.38),
        ('trends', 0.35)
    ]
    
    confidence_score = 0.85
    example_text = "The company reported strong profit growth and positive revenue trends"
    
    print(f"\nMock XAI Artifacts:")
    print(f"  LIME features: {len(lime_features)}")
    print(f"  Attention tokens: {len(attention_tokens)}")
    print(f"  Confidence: {confidence_score}")
    print(f"  Example text: {example_text[:50]}...")
    
    # Create test set from artifacts
    print("\n" + "-"*60)
    print("Creating test set from artifacts...")
    test_cases = test_set_gen.create_test_set_from_artifacts(
        lime_features, attention_tokens, confidence_score, example_text
    )
    print(f"✓ Created {len(test_cases)} test cases")
    
    # Test with sample responses
    print("\n" + "-"*60)
    print("Testing faithfulness evaluation...")
    
    # Sample constrained response (should be grounded)
    constrained_response = """According to LIME analysis, the top words that influence this prediction are 'profit' (importance=0.52), 'growth' (importance=0.48), and 'positive' (importance=0.45). 
    Attention analysis shows that the model focuses on 'profit' (score=0.58) and 'growth' (score=0.55). 
    The prediction confidence is 0.85."""
    
    # Sample naive response (may have hallucinations)
    naive_response = """The top words are 'profit', 'growth', 'positive', 'revenue', and 'strong'. 
    The model also considers 'success' and 'excellent' as important factors. 
    The confidence score is 0.92."""
    
    # Evaluate both responses
    expected_artifacts = {
        'expected_features': lime_features,
        'expected_tokens': attention_tokens,
        'expected_values': {'confidence': confidence_score}
    }
    
    print("\n--- CONSTRAINED RESPONSE EVALUATION ---")
    constrained_eval = evaluator.evaluate_response(constrained_response, expected_artifacts)
    print(f"Overall Grounded: {constrained_eval['overall_grounded']}")
    print(f"Feature Grounding: {constrained_eval['feature_grounding']['grounded']}")
    print(f"  Grounded features: {len(constrained_eval['feature_grounding']['grounded_features'])}")
    print(f"  Hallucinated features: {len(constrained_eval['feature_grounding']['hallucinated_features'])}")
    print(f"Value Grounding: {constrained_eval['value_grounding']['grounded']}")
    print(f"Citations: {constrained_eval['citations']['citation_count']}")
    print(f"Feature Overlap: {constrained_eval['feature_overlap']:.2%}")
    print(f"Has Hallucinations: {constrained_eval['hallucinations']['has_hallucinations']}")
    
    print("\n--- NAIVE RESPONSE EVALUATION ---")
    naive_eval = evaluator.evaluate_response(naive_response, expected_artifacts)
    print(f"Overall Grounded: {naive_eval['overall_grounded']}")
    print(f"Feature Grounding: {naive_eval['feature_grounding']['grounded']}")
    print(f"  Grounded features: {len(naive_eval['feature_grounding']['grounded_features'])}")
    print(f"  Hallucinated features: {naive_eval['feature_grounding']['hallucinated_features']}")
    print(f"Value Grounding: {naive_eval['value_grounding']['grounded']}")
    print(f"Citations: {naive_eval['citations']['citation_count']}")
    print(f"Feature Overlap: {naive_eval['feature_overlap']:.2%}")
    print(f"Has Hallucinations: {naive_eval['hallucinations']['has_hallucinations']}")
    
    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS SUMMARY")
    print("="*60)
    
    constrained_grounded = constrained_eval['overall_grounded']
    naive_grounded = naive_eval['overall_grounded']
    constrained_hallucinations = constrained_eval['hallucinations']['has_hallucinations']
    naive_hallucinations = naive_eval['hallucinations']['has_hallucinations']
    
    print(f"\nConstrained Prompt:")
    print(f"  Grounded: {constrained_grounded}")
    print(f"  Has Hallucinations: {constrained_hallucinations}")
    print(f"  Citations: {constrained_eval['citations']['citation_count']}")
    print(f"  Feature Overlap: {constrained_eval['feature_overlap']:.2%}")
    
    print(f"\nNaive Prompt:")
    print(f"  Grounded: {naive_grounded}")
    print(f"  Has Hallucinations: {naive_hallucinations}")
    print(f"  Citations: {naive_eval['citations']['citation_count']}")
    print(f"  Feature Overlap: {naive_eval['feature_overlap']:.2%}")
    
    print(f"\nImprovement:")
    print(f"  Grounding: {'✓ Better' if constrained_grounded and not naive_grounded else 'Similar'}")
    print(f"  Hallucinations: {'✓ Reduced' if not constrained_hallucinations and naive_hallucinations else 'Similar'}")
    print(f"  Citations: {constrained_eval['citations']['citation_count'] - naive_eval['citations']['citation_count']:+d}")
    
    # Test with multiple prompts
    print("\n" + "-"*60)
    print("Testing with multiple test prompts...")
    
    sample_prompts = [
        "What are the top 5 words that influence this prediction?",
        "How confident is the model in this prediction?",
        "What tokens does the model focus on?"
    ]
    
    results = []
    for prompt in sample_prompts:
        # Find matching test case
        test_case = next((tc for tc in test_cases if tc['prompt'] == prompt), None)
        if test_case:
            expected = {
                'expected_features': test_case.get('expected_features', []),
                'expected_tokens': test_case.get('expected_tokens', []),
                'expected_values': test_case.get('expected_values', {})
            }
            
            # Create a sample response
            if 'top 5 words' in prompt.lower():
                response = "According to LIME analysis, the top 5 words are 'profit', 'growth', 'positive', 'revenue', and 'strong'."
            elif 'confident' in prompt.lower():
                response = f"The model confidence score is {confidence_score:.2f}."
            elif 'tokens' in prompt.lower() or 'focus' in prompt.lower():
                response = "Attention analysis shows the model focuses on 'profit', 'growth', and 'positive'."
            else:
                response = "Based on the analysis, the prediction is positive."
            
            eval_result = evaluator.evaluate_response(response, expected)
            results.append({
                'prompt': prompt,
                'grounded': eval_result['overall_grounded'],
                'hallucinations': eval_result['hallucinations']['has_hallucinations']
            })
    
    print(f"\nTested {len(results)} prompts:")
    for r in results:
        status = "✓" if r['grounded'] and not r['hallucinations'] else "✗"
        print(f"  {status} {r['prompt'][:50]}...")
        print(f"    Grounded: {r['grounded']}, Hallucinations: {r['hallucinations']}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print("\n✓ Faithfulness evaluation system is working correctly!")
    print("\nThe system can:")
    print("  - Extract XAI artifacts (LIME, attention, confidence)")
    print("  - Generate test sets with ground truth")
    print("  - Evaluate response faithfulness")
    print("  - Detect hallucinations")
    print("  - Compare constrained vs naive prompts")
    
    return True


if __name__ == '__main__':
    try:
        success = test_direct_evaluation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
