#!/usr/bin/env python3
"""
Test script for interactive plot generation
This script tests the plot generator functionality independently
"""

import sys
import os

# Add xai_service to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'xai_service'))

import pandas as pd
from plot_generators.registry import PlotGeneratorRegistry

def test_plot_generator():
    """Test the sentiment plot generator"""
    print("Testing Interactive Plot Generation System")
    print("=" * 50)
    
    # Create sample sentiment data
    print("\n1. Creating sample sentiment data...")
    sample_data = {
        'title': [
            'Stock Market Rises',
            'Company Reports Loss',
            'Positive Earnings Surprise',
            'Market Volatility Increases',
            'Strong Quarterly Results'
        ],
        'asset': ['AAPL', 'MSFT', 'GOOGL', 'AAPL', 'MSFT'],
        'sentiment': [0.8, -0.6, 0.9, -0.3, 0.7],
        'date': pd.date_range('2025-01-01', periods=5, freq='D')
    }
    df = pd.DataFrame(sample_data)
    print(f"   Created DataFrame with {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Get plot generator
    print("\n2. Getting plot generator...")
    try:
        generator = PlotGeneratorRegistry.get_generator('sentiment')
        print(f"   ✓ Generator obtained: {type(generator).__name__}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Validate data
    print("\n3. Validating data...")
    is_valid, error_msg = generator.validate_data(df)
    if is_valid:
        print("   ✓ Data validation passed")
    else:
        print(f"   ✗ Data validation failed: {error_msg}")
        return False
    
    # Test plot generation
    print("\n4. Testing plot generation...")
    test_queries = [
        "Show sentiment distribution",
        "Plot sentiment over time",
        "Compare sentiment by asset"
    ]
    
    for query in test_queries:
        print(f"\n   Testing query: '{query}'")
        try:
            context = {
                'data_type': 'sentiment',
                'user_id': 'test_user'
            }
            result = generator.generate_plot(query, df, 'test_user', context)
            
            if 'plot_html' in result:
                print(f"   ✓ Plot generated successfully")
                print(f"   - Plot type: {result.get('plot_type', 'unknown')}")
                print(f"   - HTML length: {len(result['plot_html'])} characters")
            else:
                print(f"   ✗ Plot generation failed - no plot_html in result")
                return False
                
        except Exception as e:
            print(f"   ✗ Error generating plot: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    return True

if __name__ == '__main__':
    success = test_plot_generator()
    sys.exit(0 if success else 1)

