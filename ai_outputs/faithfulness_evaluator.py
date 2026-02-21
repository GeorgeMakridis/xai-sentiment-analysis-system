"""
Faithfulness Evaluator for RAG-Generated XAI Explanations

This module provides functionality to:
1. Extract structured XAI artifacts (LIME features, attention tokens, confidence scores) from stored data
2. Evaluate faithfulness of RAG responses by checking grounding in XAI artifacts
3. Detect hallucinations and compute faithfulness metrics
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class FaithfulnessEvaluator:
    """Evaluates faithfulness of RAG responses to XAI artifacts"""
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize the faithfulness evaluator
        
        Args:
            tolerance: Tolerance for numeric value matching (default: 0.01)
        """
        self.tolerance = tolerance
        
    def extract_lime_features(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract LIME top features from stored documents
        
        Args:
            docs: List of documents from vector DB search
            
        Returns:
            Dictionary with 'features' (list of (word, score) tuples) and 'source'
        """
        lime_features = []
        
        for doc in docs:
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')
            
            # Check if metadata has structured LIME features
            if 'lime_features' in metadata:
                lime_features = metadata['lime_features']
                if isinstance(lime_features, list):
                    return {'features': lime_features, 'source': 'metadata'}
            
            # Try to extract from text patterns
            # Pattern 1: "Top Important Tokens:" or "LIME highlights"
            if 'lime' in text.lower() or 'top important' in text.lower():
                # Look for patterns like "word (score)" or "word: score"
                patterns = [
                    r"'([^']+)'\s*\(score:\s*([\d.]+)\)",  # 'word' (score: 0.5)
                    r"'([^']+)'\s*\(importance[=:]\s*([\d.]+)\)",  # 'word' (importance=0.5)
                    r"([a-zA-Z]+)\s*\(([\d.]+)\)",  # word (0.5)
                    r"([a-zA-Z]+):\s*([\d.]+)",  # word: 0.5
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            if len(match) == 2:
                                word, score_str = match
                                try:
                                    score = float(score_str)
                                    lime_features.append((word.strip(), score))
                                except ValueError:
                                    continue
                        if lime_features:
                            break
            
            # Pattern 2: Check metadata for attention_analysis with LIME info
            if metadata.get('doc_type') == 'attention_analysis':
                insights = metadata.get('insights', {})
                if 'lime_features' in insights:
                    lime_features = insights['lime_features']
                    if isinstance(lime_features, list):
                        return {'features': lime_features, 'source': 'insights'}
        
        # Remove duplicates while preserving order
        seen = set()
        unique_features = []
        for word, score in lime_features:
            if word.lower() not in seen:
                seen.add(word.lower())
                unique_features.append((word, score))
        
        return {'features': unique_features[:10], 'source': 'parsed_text'}
    
    def extract_attention_tokens(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract attention top tokens from stored documents
        
        Args:
            docs: List of documents from vector DB search
            
        Returns:
            Dictionary with 'tokens' (list of (token, score) tuples) and 'source'
        """
        attention_tokens = []
        
        for doc in docs:
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')
            
            # Check if metadata has structured attention tokens
            if 'attention_tokens' in metadata:
                attention_tokens = metadata['attention_tokens']
                if isinstance(attention_tokens, list):
                    return {'tokens': attention_tokens, 'source': 'metadata'}
            
            # Check insights in metadata
            if metadata.get('doc_type') == 'attention_analysis':
                insights = metadata.get('insights', {})
                top_tokens = insights.get('top_tokens', [])
                if top_tokens:
                    if isinstance(top_tokens[0], (list, tuple)) and len(top_tokens[0]) == 2:
                        return {'tokens': top_tokens[:10], 'source': 'insights'}
            
            # Try to extract from text patterns
            # Pattern: "Top Important Tokens:" followed by numbered list
            if 'top important tokens' in text.lower() or 'attention' in text.lower():
                # Look for patterns like "1. 'token' (score: 0.5)"
                patterns = [
                    r"\d+\.\s*'([^']+)'\s*\(score:\s*([\d.]+)\)",  # 1. 'token' (score: 0.5)
                    r"\d+\.\s*'([^']+)'\s*\(([\d.]+)\)",  # 1. 'token' (0.5)
                    r"'([^']+)'\s*\(score:\s*([\d.]+)\)",  # 'token' (score: 0.5)
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            if len(match) == 2:
                                token, score_str = match
                                try:
                                    score = float(score_str)
                                    attention_tokens.append((token.strip(), score))
                                except ValueError:
                                    continue
                        if attention_tokens:
                            break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token, score in attention_tokens:
            if token.lower() not in seen:
                seen.add(token.lower())
                unique_tokens.append((token, score))
        
        return {'tokens': unique_tokens[:10], 'source': 'parsed_text'}
    
    def extract_confidence_scores(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract confidence scores and calibration metrics from stored documents
        
        Args:
            docs: List of documents from vector DB search
            
        Returns:
            Dictionary with 'confidence', 'calibrated', 'ece', etc.
        """
        confidence_data = {
            'confidence': None,
            'calibrated': False,
            'ece': None,
            'brier_score': None
        }
        
        for doc in docs:
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')
            
            # Check metadata for structured confidence data
            if 'confidence_score' in metadata:
                confidence_data['confidence'] = float(metadata['confidence_score'])
            if 'calibrated' in metadata:
                confidence_data['calibrated'] = bool(metadata['calibrated'])
            if 'ece' in metadata:
                confidence_data['ece'] = float(metadata['ece'])
            if 'brier_score' in metadata:
                confidence_data['brier_score'] = float(metadata['brier_score'])
            
            # Try to extract from text patterns
            # Pattern: "Confidence: 0.85" or "confidence score: 0.85"
            confidence_patterns = [
                r"confidence[:\s]+([\d.]+)",
                r"confidence\s+score[:\s]+([\d.]+)",
                r"prediction\s+confidence[:\s]+([\d.]+)",
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        confidence_data['confidence'] = float(match.group(1))
                        break
                    except ValueError:
                        continue
            
            # Pattern: "ECE: 0.05" or "Expected Calibration Error: 0.05"
            ece_patterns = [
                r"ece[:\s]+([\d.]+)",
                r"expected\s+calibration\s+error[:\s]+([\d.]+)",
            ]
            
            for pattern in ece_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        confidence_data['ece'] = float(match.group(1))
                        break
                    except ValueError:
                        continue
        
        return confidence_data
    
    def extract_mentioned_features(self, response: str) -> List[str]:
        """
        Extract all words/features mentioned in a RAG response
        
        Args:
            response: RAG-generated response text
            
        Returns:
            List of mentioned words/features (normalized to lowercase)
        """
        mentioned = []
        
        # Pattern 1: Words in quotes
        quoted_words = re.findall(r"'([^']+)'", response)
        mentioned.extend([w.lower().strip() for w in quoted_words])
        
        # Pattern 2: Words after "word", "words", "feature", "features"
        patterns = [
            r"(?:word|words|feature|features|token|tokens)[:\s]+([a-zA-Z]+)",
            r"top\s+\d+\s+(?:word|words|feature|features)[:\s]+([a-zA-Z]+)",
            r"most\s+important\s+(?:word|words)[:\s]+([a-zA-Z]+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            mentioned.extend([m.lower().strip() for m in matches])
        
        # Pattern 3: Listed items (bullet points, numbered lists)
        list_patterns = [
            r"[-•]\s*([a-zA-Z]+)",  # - word
            r"\d+\.\s*([a-zA-Z]+)",  # 1. word
        ]
        
        for pattern in list_patterns:
            matches = re.findall(pattern, response)
            mentioned.extend([m.lower().strip() for m in matches])
        
        # Remove duplicates and common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                     'would', 'should', 'could', 'may', 'might', 'must', 'can',
                     'this', 'that', 'these', 'those', 'and', 'or', 'but', 'if',
                     'then', 'else', 'when', 'where', 'why', 'how', 'what', 'which'}
        
        unique_mentioned = []
        seen = set()
        for word in mentioned:
            word_lower = word.lower().strip()
            if word_lower and word_lower not in stop_words and word_lower not in seen:
                seen.add(word_lower)
                unique_mentioned.append(word_lower)
        
        return unique_mentioned
    
    def extract_mentioned_values(self, response: str) -> List[float]:
        """
        Extract all numeric values mentioned in a RAG response
        
        Args:
            response: RAG-generated response text
            
        Returns:
            List of mentioned numeric values
        """
        values = []
        
        # Pattern: Decimal numbers (0.85, 0.5, etc.)
        decimal_pattern = r"\b([\d]+\.[\d]+)\b"
        matches = re.findall(decimal_pattern, response)
        
        for match in matches:
            try:
                value = float(match)
                # Only include values that look like probabilities/scores (0-1 range)
                if 0.0 <= value <= 1.0:
                    values.append(value)
            except ValueError:
                continue
        
        return values
    
    def extract_citations(self, response: str) -> List[str]:
        """
        Extract citations from RAG response
        
        Args:
            response: RAG-generated response text
            
        Returns:
            List of citation phrases found
        """
        citations = []
        
        citation_patterns = [
            r"according\s+to\s+([^,\.]+)",
            r"based\s+on\s+([^,\.]+)",
            r"([^,\.]+)\s+analysis\s+shows",
            r"([^,\.]+)\s+shows",
            r"([^,\.]+)\s+indicates",
            r"per\s+([^,\.]+)",
            r"from\s+([^,\.]+)",
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            citations.extend([m.strip() for m in matches])
        
        # Filter for relevant citations (LIME, attention, etc.)
        relevant_citations = []
        keywords = ['lime', 'attention', 'model', 'analysis', 'prediction', 'confidence']
        
        for citation in citations:
            citation_lower = citation.lower()
            if any(keyword in citation_lower for keyword in keywords):
                relevant_citations.append(citation)
        
        return relevant_citations
    
    def check_feature_grounding(self, response: str, expected_features: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Check if all mentioned features in response are grounded in expected features
        
        Args:
            response: RAG-generated response
            expected_features: List of (word, score) tuples from LIME/attention
            
        Returns:
            Dictionary with grounding results
        """
        mentioned = self.extract_mentioned_features(response)
        expected_words = {word.lower() for word, _ in expected_features}
        
        grounded = []
        hallucinated = []
        
        for word in mentioned:
            if word in expected_words:
                grounded.append(word)
            else:
                hallucinated.append(word)
        
        all_grounded = len(hallucinated) == 0
        grounding_ratio = len(grounded) / len(mentioned) if mentioned else 0.0
        
        return {
            'grounded': all_grounded,
            'grounding_ratio': grounding_ratio,
            'mentioned_features': mentioned,
            'grounded_features': grounded,
            'hallucinated_features': hallucinated,
            'total_mentioned': len(mentioned),
            'total_grounded': len(grounded),
            'total_hallucinated': len(hallucinated)
        }
    
    def check_value_grounding(self, response: str, expected_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if all mentioned numeric values match expected values
        
        Args:
            response: RAG-generated response
            expected_values: Dictionary with expected values (e.g., {'confidence': 0.85})
            
        Returns:
            Dictionary with value grounding results
        """
        mentioned = self.extract_mentioned_values(response)
        grounded = []
        hallucinated = []
        
        for value in mentioned:
            matched = False
            for key, expected_val in expected_values.items():
                if expected_val is not None:
                    if abs(value - expected_val) <= self.tolerance:
                        grounded.append((key, value, expected_val))
                        matched = True
                        break
            
            if not matched:
                hallucinated.append(value)
        
        all_grounded = len(hallucinated) == 0
        grounding_ratio = len(grounded) / len(mentioned) if mentioned else 0.0
        
        return {
            'grounded': all_grounded,
            'grounding_ratio': grounding_ratio,
            'mentioned_values': mentioned,
            'grounded_values': grounded,
            'hallucinated_values': hallucinated,
            'total_mentioned': len(mentioned),
            'total_grounded': len(grounded),
            'total_hallucinated': len(hallucinated)
        }
    
    def check_citation_requirements(self, response: str) -> Dict[str, Any]:
        """
        Check if response includes citations
        
        Args:
            response: RAG-generated response
            
        Returns:
            Dictionary with citation results
        """
        citations = self.extract_citations(response)
        
        return {
            'has_citations': len(citations) > 0,
            'citation_count': len(citations),
            'citations': citations
        }
    
    def detect_hallucinations(self, response: str, expected_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combined hallucination detection using all checks
        
        Args:
            response: RAG-generated response
            expected_artifacts: Dictionary with expected_features, expected_tokens, expected_values
            
        Returns:
            Dictionary with comprehensive hallucination detection results
        """
        results = {
            'has_hallucinations': False,
            'hallucination_count': 0,
            'details': []
        }
        
        # Check feature grounding
        expected_features = expected_artifacts.get('expected_features', [])
        if expected_features:
            feature_check = self.check_feature_grounding(response, expected_features)
            if feature_check['total_hallucinated'] > 0:
                results['has_hallucinations'] = True
                results['hallucination_count'] += feature_check['total_hallucinated']
                results['details'].append({
                    'type': 'feature_hallucination',
                    'hallucinated_features': feature_check['hallucinated_features']
                })
        
        # Check value grounding
        expected_values = expected_artifacts.get('expected_values', {})
        if expected_values:
            value_check = self.check_value_grounding(response, expected_values)
            if value_check['total_hallucinated'] > 0:
                results['has_hallucinations'] = True
                results['hallucination_count'] += value_check['total_hallucinated']
                results['details'].append({
                    'type': 'value_hallucination',
                    'hallucinated_values': value_check['hallucinated_values']
                })
        
        # Check for unsupported causal claims (simple heuristic)
        causal_patterns = [
            r"because\s+of\s+([^,\.]+)",
            r"caused\s+by\s+([^,\.]+)",
            r"due\s+to\s+([^,\.]+)",
        ]
        
        for pattern in causal_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # Check if the cause is mentioned in expected features
                for match in matches:
                    cause_words = match.lower().split()
                    if expected_features:
                        expected_words = {word.lower() for word, _ in expected_features}
                        for cause_word in cause_words:
                            if cause_word not in expected_words and len(cause_word) > 3:
                                results['has_hallucinations'] = True
                                results['hallucination_count'] += 1
                                results['details'].append({
                                    'type': 'unsupported_causal_claim',
                                    'claim': match
                                })
                                break
        
        return results
    
    def compute_feature_overlap(self, response: str, expected_features: List[Tuple[str, float]]) -> float:
        """
        Compute Jaccard similarity between mentioned and expected features
        
        Args:
            response: RAG-generated response
            expected_features: List of (word, score) tuples
            
        Returns:
            Jaccard similarity score (0-1)
        """
        mentioned = set(self.extract_mentioned_features(response))
        expected = {word.lower() for word, _ in expected_features}
        
        if not mentioned and not expected:
            return 1.0
        if not mentioned or not expected:
            return 0.0
        
        intersection = mentioned & expected
        union = mentioned | expected
        
        return len(intersection) / len(union) if union else 0.0
    
    def evaluate_response(self, response: str, expected_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single RAG response
        
        Args:
            response: RAG-generated response
            expected_artifacts: Dictionary with expected_features, expected_tokens, expected_values
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            'response': response,
            'feature_grounding': {},
            'value_grounding': {},
            'citations': {},
            'hallucinations': {},
            'feature_overlap': 0.0,
            'overall_grounded': False
        }
        
        # Feature grounding check
        expected_features = expected_artifacts.get('expected_features', [])
        if expected_features:
            results['feature_grounding'] = self.check_feature_grounding(response, expected_features)
            results['feature_overlap'] = self.compute_feature_overlap(response, expected_features)
        
        # Value grounding check
        expected_values = expected_artifacts.get('expected_values', {})
        if expected_values:
            results['value_grounding'] = self.check_value_grounding(response, expected_values)
        
        # Citation check
        results['citations'] = self.check_citation_requirements(response)
        
        # Hallucination detection
        results['hallucinations'] = self.detect_hallucinations(response, expected_artifacts)
        
        # Overall grounded: all checks pass
        feature_ok = results['feature_grounding'].get('grounded', True) if expected_features else True
        value_ok = results['value_grounding'].get('grounded', True) if expected_values else True
        no_hallucinations = not results['hallucinations'].get('has_hallucinations', False)
        
        results['overall_grounded'] = feature_ok and value_ok and no_hallucinations
        
        return results
