import unittest
import pandas as pd

from Utils import *

class TestUtils(unittest.TestCase):
    
    def test_read_data_return_dataframe(self):
        df = read_data_return_dataframe('testdata.txt')

        self.assertEqual(pd.DataFrame, type(df))
        self.assertEqual(20, df.size)
  
    def test_split_df_labels_attributes(self):
        labels, attrs = split_df_labels_attributes(read_data_return_dataframe('testdata.txt'))

        self.assertEqual(labels.columns, ['Label'])
        self.assertEqual(['no   ', 'no   '], labels['Label'].tolist())
        self.assertEqual('Label' not in attrs.columns, True)
        
    def test_split_df_to_train_test_dfs(self):
        df = read_data_return_dataframe('testdata-alt.txt')
        train, test = split_df_to_train_test_dfs(df, upper=.7)

        self.assertEqual(train.index.equals(test.index), False)
        self.assertEqual(len(train) + len(test), len(df))
        self.assertNotEqual(train.values, test.values)
        
        
if __name__ == '__main__':
    unittest.main()