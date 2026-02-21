"""
Unit tests for faithfulness evaluator

Run with: python -m pytest tests/test_faithfulness_evaluator.py
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_outputs.faithfulness_evaluator import FaithfulnessEvaluator


class TestFaithfulnessEvaluator:
    """Test cases for FaithfulnessEvaluator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.evaluator = FaithfulnessEvaluator(tolerance=0.01)
    
    def test_extract_mentioned_features(self):
        """Test extraction of mentioned features from response"""
        response = "The top words are 'profit', 'growth', and 'revenue'. The model focuses on 'positive' sentiment."
        
        mentioned = self.evaluator.extract_mentioned_features(response)
        
        assert 'profit' in mentioned
        assert 'growth' in mentioned
        assert 'revenue' in mentioned
        assert 'positive' in mentioned
        assert 'sentiment' in mentioned
    
    def test_extract_mentioned_values(self):
        """Test extraction of numeric values from response"""
        response = "The confidence score is 0.85. The importance is 0.75."
        
        values = self.evaluator.extract_mentioned_values(response)
        
        assert 0.85 in values
        assert 0.75 in values
        assert len(values) == 2
    
    def test_extract_citations(self):
        """Test extraction of citations from response"""
        response = "According to LIME analysis, the top features are important. Attention analysis shows high scores."
        
        citations = self.evaluator.extract_citations(response)
        
        assert len(citations) > 0
        assert any('lime' in c.lower() for c in citations)
        assert any('attention' in c.lower() for c in citations)
    
    def test_check_feature_grounding(self):
        """Test feature grounding check"""
        response = "The top words are 'profit' and 'growth'."
        expected_features = [('profit', 0.5), ('growth', 0.4), ('revenue', 0.3)]
        
        result = self.evaluator.check_feature_grounding(response, expected_features)
        
        assert result['grounded'] == True
        assert len(result['grounded_features']) == 2
        assert len(result['hallucinated_features']) == 0
    
    def test_check_feature_grounding_with_hallucination(self):
        """Test feature grounding with hallucinated features"""
        response = "The top words are 'profit', 'growth', and 'nonexistent'."
        expected_features = [('profit', 0.5), ('growth', 0.4)]
        
        result = self.evaluator.check_feature_grounding(response, expected_features)
        
        assert result['grounded'] == False
        assert 'nonexistent' in result['hallucinated_features']
    
    def test_check_value_grounding(self):
        """Test value grounding check"""
        response = "The confidence score is 0.85."
        expected_values = {'confidence': 0.85}
        
        result = self.evaluator.check_value_grounding(response, expected_values)
        
        assert result['grounded'] == True
        assert len(result['grounded_values']) > 0
    
    def test_check_value_grounding_with_tolerance(self):
        """Test value grounding with tolerance"""
        response = "The confidence score is 0.851."
        expected_values = {'confidence': 0.85}
        
        result = self.evaluator.check_value_grounding(response, expected_values)
        
        # Should match within tolerance (0.01)
        assert result['grounded'] == True
    
    def test_check_citation_requirements(self):
        """Test citation requirements check"""
        response = "According to LIME analysis, the features are important."
        
        result = self.evaluator.check_citation_requirements(response)
        
        assert result['has_citations'] == True
        assert result['citation_count'] > 0
    
    def test_detect_hallucinations(self):
        """Test hallucination detection"""
        response = "The top words are 'profit' and 'growth'. The confidence is 0.85."
        expected_artifacts = {
            'expected_features': [('profit', 0.5), ('growth', 0.4)],
            'expected_values': {'confidence': 0.85}
        }
        
        result = self.evaluator.detect_hallucinations(response, expected_artifacts)
        
        assert result['has_hallucinations'] == False
        assert result['hallucination_count'] == 0
    
    def test_detect_hallucinations_with_hallucination(self):
        """Test hallucination detection with actual hallucinations"""
        response = "The top words are 'profit', 'growth', and 'fake_word'. The confidence is 0.95."
        expected_artifacts = {
            'expected_features': [('profit', 0.5), ('growth', 0.4)],
            'expected_values': {'confidence': 0.85}
        }
        
        result = self.evaluator.detect_hallucinations(response, expected_artifacts)
        
        assert result['has_hallucinations'] == True
        assert result['hallucination_count'] > 0
    
    def test_compute_feature_overlap(self):
        """Test feature overlap computation"""
        response = "The top words are 'profit' and 'growth'."
        expected_features = [('profit', 0.5), ('growth', 0.4), ('revenue', 0.3)]
        
        overlap = self.evaluator.compute_feature_overlap(response, expected_features)
        
        assert 0.0 <= overlap <= 1.0
        assert overlap > 0.0  # Should have some overlap
    
    def test_evaluate_response_comprehensive(self):
        """Test comprehensive response evaluation"""
        response = "According to LIME analysis, the top words are 'profit' and 'growth'. The confidence score is 0.85."
        expected_artifacts = {
            'expected_features': [('profit', 0.5), ('growth', 0.4)],
            'expected_values': {'confidence': 0.85}
        }
        
        result = self.evaluator.evaluate_response(response, expected_artifacts)
        
        assert 'feature_grounding' in result
        assert 'value_grounding' in result
        assert 'citations' in result
        assert 'hallucinations' in result
        assert 'feature_overlap' in result
        assert 'overall_grounded' in result
    
    def test_extract_lime_features_from_metadata(self):
        """Test LIME feature extraction from metadata"""
        docs = [
            {
                'text': 'Some text',
                'metadata': {
                    'lime_features': [('profit', 0.5), ('growth', 0.4)]
                }
            }
        ]
        
        result = self.evaluator.extract_lime_features(docs)
        
        assert len(result['features']) == 2
        assert result['source'] == 'metadata'
    
    def test_extract_attention_tokens_from_metadata(self):
        """Test attention token extraction from metadata"""
        docs = [
            {
                'text': 'Some text',
                'metadata': {
                    'attention_tokens': [('token1', 0.3), ('token2', 0.2)]
                }
            }
        ]
        
        result = self.evaluator.extract_attention_tokens(docs)
        
        assert len(result['tokens']) == 2
        assert result['source'] == 'metadata'
    
    def test_extract_confidence_scores_from_metadata(self):
        """Test confidence score extraction from metadata"""
        docs = [
            {
                'text': 'Some text',
                'metadata': {
                    'confidence_score': 0.85
                }
            }
        ]
        
        result = self.evaluator.extract_confidence_scores(docs)
        
        assert result['confidence'] == 0.85


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
