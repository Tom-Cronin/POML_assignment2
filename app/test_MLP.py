import unittest
from MLP import MLP
from Utils import *
from Layer import Layer

class TestMLP(unittest.TestCase):
    
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
        self.assertEqual(type(MLP()), MLP)

    def test_add_layer(self):
        X, x_t, y, y_t = self.setup()
        m, n = X.shape
        mlp = MLP()
        mlp.add_layer(output_size = m, activation='relu', input_size=n) # Add a layer of 32 inputs
        mlp.add_layer(output_size = 1, activation='sigmoid', input_size= m)
        self.assertEqual(len(mlp.layers), 2)
        self.assertEqual(type(mlp.layers[0]), Layer)
    
    def test_train(self):
        X, x_t, y, y_t = self.setup()
        m,n = X.shape
        mlp = MLP()
        mlp.add_layer(output_size = m, activation='relu', input_size=n) # Add a layer of 32 inputs
        mlp.add_layer(output_size = 1, activation='sigmoid', input_size= m)
        Z = mlp.train(X=X, Y=y, iters= 1000)
        self.assertListEqual(Z.tolist(), [[1,1,1]])
        
if __name__ == '__main__':
    unittest.main()
