"""
Unit tests for image data exploration plots
Tests data-only plots (not model performance/XAI plots)
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'xai_service'))

from plot_generators.image_plot_generator import ImagePlotGenerator


class TestImageDataPlots(unittest.TestCase):
    """Test data exploration plots for image data"""
    
    def setUp(self):
        """Set up test data"""
        self.generator = ImagePlotGenerator()
        
        # Create sample MNIST-like data
        np.random.seed(42)
        n_samples = 100
        
        # Create base64-like image data (simplified)
        image_data_list = [f"data:image/png;base64,test{i}" for i in range(n_samples)]
        
        # Create labels (0-9)
        labels = np.random.randint(0, 10, n_samples)
        
        # Create split (train/test)
        splits = ['train' if i < n_samples // 2 else 'test' for i in range(n_samples)]
        
        self.test_data = pd.DataFrame({
            'image_data': image_data_list,
            'label': labels,
            'digit': labels,  # Alias
            'split': splits,
            'image_id': [f"img_{i:03d}" for i in range(n_samples)]
        })
    
    def test_class_distribution_plot(self):
        """Test class distribution plot generation"""
        plot_spec = {
            'plot_type': 'class_distribution',
            'chart_type': 'class_distribution',
            'title': 'Test Class Distribution'
        }
        
        result = self.generator._create_classification_distribution(
            self.test_data, plot_spec, 'Test Class Distribution'
        )
        
        # Should return a Plotly figure
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'data'))
    
    def test_train_test_distribution_plot(self):
        """Test train/test split distribution plot"""
        plot_spec = {
            'plot_type': 'train_test_distribution',
            'chart_type': 'train_test_distribution',
            'title': 'Test Train/Test Split'
        }
        
        result = self.generator._create_train_test_distribution(
            self.test_data, plot_spec, 'Test Train/Test Split'
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'data'))
    
    def test_dataset_overview_plot(self):
        """Test dataset overview plot"""
        plot_spec = {
            'plot_type': 'dataset_overview',
            'chart_type': 'dataset_overview',
            'title': 'Test Dataset Overview'
        }
        
        result = self.generator._create_dataset_overview(
            self.test_data, plot_spec, 'Test Dataset Overview'
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'data'))
    
    def test_class_balance_analysis_plot(self):
        """Test class balance analysis plot"""
        plot_spec = {
            'plot_type': 'class_balance_analysis',
            'chart_type': 'class_balance_analysis',
            'title': 'Test Class Balance'
        }
        
        result = self.generator._create_class_balance_analysis(
            self.test_data, plot_spec, 'Test Class Balance'
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'data'))
    
    def test_image_grid_by_class(self):
        """Test image grid by class"""
        plot_spec = {
            'plot_type': 'image_grid_by_class',
            'chart_type': 'image_grid_by_class',
            'title': 'Test Image Grid'
        }
        
        result = self.generator._create_image_grid_by_class(
            self.test_data, plot_spec, 'Test Image Grid'
        )
        
        # Should return dict with HTML or Figure
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, dict) or hasattr(result, 'data'))
    
    def test_plot_summary_metadata(self):
        """Test that plot summaries include data-focused metadata"""
        plot_spec = {
            'plot_type': 'class_distribution',
            'chart_type': 'class_distribution',
            'title': 'Test Plot'
        }
        
        summary = self.generator._build_plot_summary(plot_spec, self.test_data)
        
        # Check summary structure
        self.assertIn('title', summary)
        self.assertIn('data', summary)
        self.assertIn('summary_text', summary)
        
        # Check data-focused metadata
        summary_data = summary['data']
        self.assertIn('row_count', summary_data)
        self.assertIn('plot_type', summary_data)
        
        # For class distribution, should have class counts
        if summary_data.get('plot_type') == 'class_distribution':
            self.assertIn('class_counts', summary_data)
    
    def test_fallback_interpretation_data_queries(self):
        """Test that fallback interpretation recognizes data exploration queries"""
        # Test class distribution query
        result = self.generator._fallback_interpretation(
            "Show class distribution", self.test_data
        )
        self.assertEqual(result.get('chart_type'), 'class_distribution')
        
        # Test train/test query
        result = self.generator._fallback_interpretation(
            "Show train test distribution", self.test_data
        )
        self.assertEqual(result.get('chart_type'), 'train_test_distribution')
        
        # Test dataset overview query
        result = self.generator._fallback_interpretation(
            "Show dataset overview", self.test_data
        )
        self.assertEqual(result.get('chart_type'), 'dataset_overview')
        
        # Test class balance query
        result = self.generator._fallback_interpretation(
            "Show class balance", self.test_data
        )
        self.assertEqual(result.get('chart_type'), 'class_balance_analysis')
    
    def test_data_validation(self):
        """Test that data validation works for image data"""
        is_valid, error = self.generator.validate_data(self.test_data)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns gracefully"""
        # Data without label column
        data_no_label = self.test_data.drop(columns=['label', 'digit'])
        
        plot_spec = {
            'plot_type': 'class_distribution',
            'chart_type': 'class_distribution',
            'title': 'Test'
        }
        
        # Should not crash, should return fallback plot
        result = self.generator._create_classification_distribution(
            data_no_label, plot_spec, 'Test'
        )
        self.assertIsNotNone(result)
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_data = pd.DataFrame(columns=['image_data', 'label'])
        
        is_valid, error = self.generator.validate_data(empty_data)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)


if __name__ == '__main__':
    unittest.main()
