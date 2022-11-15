import unittest

from Layer import Layer
from Utils import *

class TestLayer(unittest.TestCase):
    def setup(self):
        wildfires = read_data_return_dataframe("testdata-alt.txt")
        # Copy to be used for the rest of the assignment
        wildfires_copy = wildfires.copy()
        # wildfires_copy = convert_label(wildfires,'fire',['no', 'yes'],[0, 1])


        features = ['year', 'temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'day', 'month', 'wind_speed']
        X_train, X_test, y_train, y_test = split_df_to_train_test_dfs(wildfires_copy, test_set_size=.1,
                                                    random_state=42)
        X_train = X_train[features].values  # returns a numpy NdArray of the features
        X_test = X_test[features].values  # returns a numpy NdArray of the features
        X_train = Normalize(X_train, features)
        X_test = Normalize(X_test, features)

        X_train = np.asarray(X_train)[0:3]
        y_train = np.asarray(y_train).flatten()
        y_train = np.asarray([1 if 'yes' in y else 0 for y in y_train])[0:3]
        return X_train, X_test, y_train, y_test
        
    
    def test_constructor(self):
        self.assertEqual(type(Layer(2, "relu", input_size=9)), Layer)
        
    def test_validate_input_params(self):
          with self.assertRaises(ValueError):
              Layer(-2, "relu", input_size=9)
          with self.assertRaises(ValueError):
              Layer(3, "relu", input_size=-8)
         
    def test_predict(self):
        X, x_t, y, y_t = self.setup()
        l1 = Layer(3, "relu", input_size=9)
        pred = l1.predict(X)

        self.assertListEqual(pred.tolist() , [[0, 0, 0],[0, 0, 0],[0, 0, 0]])
        
        
if __name__ == '__main__':
    unittest.main()