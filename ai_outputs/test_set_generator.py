"""
Test Set Generator for RAG Faithfulness Evaluation

Generates a fixed test set of 20-30 prompts with ground truth expected answers
based on XAI artifacts (LIME features, attention tokens, confidence scores).
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime


class TestSetGenerator:
    """Generate test prompts with ground truth for faithfulness evaluation"""
    
    def __init__(self):
        """Initialize test set generator"""
        self.test_prompts = []
    
    def generate_test_prompts(self) -> List[Dict[str, Any]]:
        """
        Generate fixed set of test prompts with categories
        
        Returns:
            List of test prompt dictionaries
        """
        prompts = []
        
        # Category 1: Feature Importance Questions (10 prompts)
        feature_prompts = [
            {
                'prompt': 'What are the top 5 words that influence this prediction?',
                'category': 'feature_importance',
                'expected_artifact_type': 'lime_features',
                'expected_count': 5
            },
            {
                'prompt': 'Which words are most important for the positive sentiment?',
                'category': 'feature_importance',
                'expected_artifact_type': 'lime_features',
                'expected_count': 10
            },
            {
                'prompt': 'Explain why this text is classified as negative',
                'category': 'feature_importance',
                'expected_artifact_type': 'lime_features',
                'expected_count': 10
            },
            {
                'prompt': 'What are the most important features for this prediction?',
                'category': 'feature_importance',
                'expected_artifact_type': 'lime_features',
                'expected_count': 10
            },
            {
                'prompt': 'Which words drive the sentiment classification?',
                'category': 'feature_importance',
                'expected_artifact_type': 'lime_features',
                'expected_count': 10
            },
            {
                'prompt': 'What words contribute most to the model decision?',
                'category': 'feature_importance',
                'expected_artifact_type': 'lime_features',
                'expected_count': 10
            },
            {
                'prompt': 'List the top words that affect this prediction',
                'category': 'feature_importance',
                'expected_artifact_type': 'lime_features',
                'expected_count': 10
            },
            {
                'prompt': 'What are the key words for this sentiment analysis?',
                'category': 'feature_importance',
                'expected_artifact_type': 'lime_features',
                'expected_count': 10
            },
            {
                'prompt': 'Which features are most significant for this prediction?',
                'category': 'feature_importance',
                'expected_artifact_type': 'lime_features',
                'expected_count': 10
            },
            {
                'prompt': 'What words should I focus on to understand this prediction?',
                'category': 'feature_importance',
                'expected_artifact_type': 'lime_features',
                'expected_count': 10
            }
        ]
        
        # Category 2: Attention Questions (5 prompts)
        attention_prompts = [
            {
                'prompt': 'What tokens does the model focus on?',
                'category': 'attention',
                'expected_artifact_type': 'attention_tokens',
                'expected_count': 10
            },
            {
                'prompt': 'Which words get the most attention?',
                'category': 'attention',
                'expected_artifact_type': 'attention_tokens',
                'expected_count': 10
            },
            {
                'prompt': 'What does the attention analysis show?',
                'category': 'attention',
                'expected_artifact_type': 'attention_tokens',
                'expected_count': 10
            },
            {
                'prompt': 'Which tokens are highlighted by the attention mechanism?',
                'category': 'attention',
                'expected_artifact_type': 'attention_tokens',
                'expected_count': 10
            },
            {
                'prompt': 'What words does the transformer model pay attention to?',
                'category': 'attention',
                'expected_artifact_type': 'attention_tokens',
                'expected_count': 10
            }
        ]
        
        # Category 3: Confidence Questions (5 prompts)
        confidence_prompts = [
            {
                'prompt': 'How confident is the model in this prediction?',
                'category': 'confidence',
                'expected_artifact_type': 'confidence_score',
                'expected_count': 1
            },
            {
                'prompt': 'Should I trust this prediction?',
                'category': 'confidence',
                'expected_artifact_type': 'confidence_score',
                'expected_count': 1
            },
            {
                'prompt': 'What is the prediction confidence score?',
                'category': 'confidence',
                'expected_artifact_type': 'confidence_score',
                'expected_count': 1
            },
            {
                'prompt': 'How reliable is this sentiment prediction?',
                'category': 'confidence',
                'expected_artifact_type': 'confidence_score',
                'expected_count': 1
            },
            {
                'prompt': 'What is the model confidence level?',
                'category': 'confidence',
                'expected_artifact_type': 'confidence_score',
                'expected_count': 1
            }
        ]
        
        # Category 4: General Explanation (5 prompts)
        general_prompts = [
            {
                'prompt': 'How does the model make this prediction?',
                'category': 'general_explanation',
                'expected_artifact_type': 'mixed',  # Should mention both LIME and attention
                'expected_count': None
            },
            {
                'prompt': 'Explain the sentiment classification',
                'category': 'general_explanation',
                'expected_artifact_type': 'mixed',
                'expected_count': None
            },
            {
                'prompt': 'What factors influence this prediction?',
                'category': 'general_explanation',
                'expected_artifact_type': 'mixed',
                'expected_count': None
            },
            {
                'prompt': 'How was this sentiment determined?',
                'category': 'general_explanation',
                'expected_artifact_type': 'mixed',
                'expected_count': None
            },
            {
                'prompt': 'Can you explain this prediction?',
                'category': 'general_explanation',
                'expected_artifact_type': 'mixed',
                'expected_count': None
            }
        ]
        
        prompts.extend(feature_prompts)
        prompts.extend(attention_prompts)
        prompts.extend(confidence_prompts)
        prompts.extend(general_prompts)
        
        return prompts
    
    def create_test_set_from_artifacts(self, 
                                      lime_features: List[tuple],
                                      attention_tokens: List[tuple],
                                      confidence_score: float,
                                      example_text: str = "") -> List[Dict[str, Any]]:
        """
        Create test set with ground truth from actual XAI artifacts
        
        Args:
            lime_features: List of (word, score) tuples from LIME
            attention_tokens: List of (token, score) tuples from attention
            confidence_score: Confidence score from prediction
            example_text: Original text being analyzed (optional)
            
        Returns:
            List of test cases with ground truth
        """
        base_prompts = self.generate_test_prompts()
        test_cases = []
        
        for prompt_template in base_prompts:
            test_case = {
                'prompt': prompt_template['prompt'],
                'category': prompt_template['category'],
                'expected_artifact_type': prompt_template['expected_artifact_type'],
                'example_text': example_text,
                'expected_features': [],
                'expected_tokens': [],
                'expected_confidence': None,
                'expected_values': {}
            }
            
            # Set ground truth based on artifact type
            if prompt_template['expected_artifact_type'] == 'lime_features':
                count = prompt_template.get('expected_count', 10)
                test_case['expected_features'] = lime_features[:count]
                test_case['expected_values'] = {
                    'confidence': confidence_score
                }
            elif prompt_template['expected_artifact_type'] == 'attention_tokens':
                count = prompt_template.get('expected_count', 10)
                test_case['expected_tokens'] = attention_tokens[:count]
                test_case['expected_values'] = {
                    'confidence': confidence_score
                }
            elif prompt_template['expected_artifact_type'] == 'confidence_score':
                test_case['expected_confidence'] = confidence_score
                test_case['expected_values'] = {
                    'confidence': confidence_score
                }
            elif prompt_template['expected_artifact_type'] == 'mixed':
                # General explanation should mention both
                test_case['expected_features'] = lime_features[:10]
                test_case['expected_tokens'] = attention_tokens[:10]
                test_case['expected_confidence'] = confidence_score
                test_case['expected_values'] = {
                    'confidence': confidence_score
                }
            
            test_cases.append(test_case)
        
        return test_cases
    
    def save_test_set(self, test_cases: List[Dict[str, Any]], filepath: str):
        """
        Save test set to JSON file
        
        Args:
            test_cases: List of test case dictionaries
            filepath: Path to save JSON file
        """
        test_set = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_test_cases': len(test_cases),
                'categories': {
                    'feature_importance': sum(1 for tc in test_cases if tc['category'] == 'feature_importance'),
                    'attention': sum(1 for tc in test_cases if tc['category'] == 'attention'),
                    'confidence': sum(1 for tc in test_cases if tc['category'] == 'confidence'),
                    'general_explanation': sum(1 for tc in test_cases if tc['category'] == 'general_explanation')
                }
            },
            'test_cases': test_cases
        }
        
        with open(filepath, 'w') as f:
            json.dump(test_set, f, indent=2, default=str)
        
        print(f"Saved test set with {len(test_cases)} test cases to {filepath}")
    
    def load_test_set(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load test set from JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of test case dictionaries
        """
        with open(filepath, 'r') as f:
            test_set = json.load(f)
        
        return test_set.get('test_cases', [])


def create_default_test_set(filepath: str = 'ai_outputs/test_set.json'):
    """
    Create a default test set with placeholder ground truth
    
    This can be used as a template, then updated with actual XAI artifacts
    """
    generator = TestSetGenerator()
    prompts = generator.generate_test_prompts()
    
    # Create test cases with placeholder ground truth
    test_cases = []
    for prompt in prompts:
        test_case = {
            'prompt': prompt['prompt'],
            'category': prompt['category'],
            'expected_artifact_type': prompt['expected_artifact_type'],
            'example_text': '',
            'expected_features': [],  # Will be populated from actual XAI results
            'expected_tokens': [],  # Will be populated from actual XAI results
            'expected_confidence': None,  # Will be populated from actual XAI results
            'expected_values': {}
        }
        test_cases.append(test_case)
    
    generator.save_test_set(test_cases, filepath)
    return test_cases


if __name__ == '__main__':
    # Create default test set
    create_default_test_set('test_set.json')
    print("Default test set created. Update with actual XAI artifacts before use.")
