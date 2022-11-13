

import unittest
import pandas as pd

from utils import read_data_return_dataframe, split_df_labels_attributes

class TestUtils(unittest.TestCase):
    
    def test_read_data_return_dataframe(self):
        df = read_data_return_dataframe('testdata.txt')
        self.assertEqual(pd.DataFrame, type(df))
        self.assertEqual(20, df.size)
  
    def test_split_df_labels_attributes(self):
        labels, attrs = split_df_labels_attributes(read_data_return_dataframe('testdata.txt'))

        self.assertEqual(labels.columns, ['Label'])
        self.assertEqual(['no   ', 'no   '], labels['Label'].tolist())
        self.assertEqual('Label' not in attrs, True)
        
if __name__ == '__main__':
    unittest.main()