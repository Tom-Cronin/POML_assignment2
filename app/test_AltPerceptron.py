import unittest
import numpy as np
from AltPerceptron import Perceptron
from Utils import *


class TestPerceptron(unittest.TestCase):

    def test_constructor_validation(self):
        """Should fail validation"""
        with self.assertRaises(ValueError):
            Perceptron(n_iters=[56,1,2])
        with self.assertRaises(ValueError):
            Perceptron(learn_rate=-1)

    def test_constructor(self):
        self.assertEqual(Perceptron, type(Perceptron(learn_rate=0.1, n_iters=100)))

    def test_fit(self):
        df = read_data_return_dataframe("./testdata-alt.txt")
        _, X = split_df_labels_attributes(df)
        P = Perceptron(learn_rate=0.5, n_iters=1000).fit(X, np.array([0,0,1,0,1,1]))
        self.assertEquals(Perceptron, type(P.fit(X, np.array([0,0,1,0,1,1]))))

    def test_predict(self):
        df = read_data_return_dataframe("./testdata-alt.txt")
        _, X = split_df_labels_attributes(df)
        _, X_test = split_df_labels_attributes(read_data_return_dataframe('./testdata.txt'))
        P = Perceptron(learn_rate=0.5, n_iters=1000).fit(X, np.array([0,0,1,0,1,1]))
        self.assertListEqual(P.predict(X_test).tolist(), [1, 1])


if __name__ == '__main__':
    unittest.main()
