"""
Tests to the module features_analysis_tools.py

To run - write in the terminal:
`python -m unittest utils.tests.test_features_analysis_tools`
"""

import unittest
import pandas as pd
from utils.features_analysis_tools import (
    mark_features_with_not_null_level_lower_than,
    extract_col_With_the_highest_iv_among_correlated_features
)

class TestFeaturesAnalysisTools(unittest.TestCase):
    def test_mark_features_with_not_null_level_lower_than(self):
        # Create a sample DataFrame
        data = pd.DataFrame({
            'feature_name': ['A', 'B', 'C'],
            '%_not_nulls': [8, 60, 0.2],
            'AsFeature': ['', '', ''],
            'Comment': ['', '', '']
        }, index=['A', 'B', 'C'])
        
        # Call the function with null_level = 10
        modified_data = mark_features_with_not_null_level_lower_than(data, not_null_level=10)
        
        # Check if modifications are as expected
        self.assertEqual(modified_data.loc['C', 'AsFeature'], '-')
        self.assertEqual(modified_data.loc['C', 'Comment'], '; lower than 10% not null values;')
        
        # Check if modifications are as expected
        self.assertEqual(modified_data.loc['A', 'AsFeature'], '-')
        self.assertEqual(modified_data.loc['A', 'Comment'],  '; lower than 10% not null values;') 
        
        # Check if modifications are as expected
        self.assertEqual(modified_data.loc['B', 'AsFeature'], '')
        self.assertEqual(modified_data.loc['B', 'Comment'],  '')       

if __name__ == '__main__':
    unittest.main()
