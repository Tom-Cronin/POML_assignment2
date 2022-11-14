
    # Should fail validation
    # >>> MLP(n_iters="whoops")
    # Traceback (most recent call last):
    # ...
    # ValueError: n_iters must be an integer and a natural number

    # Should fail validation
    # >>> MLP(hidden_layer_size=(0,0))
    # Traceback (most recent call last):
    # ...
    # ValueError: hidden_layer_size must contain natural numbers of type int

    # Should fail validation
    # >>> MLP(learn_rate=0)
    # Traceback (most recent call last):
    # ...
    # ValueError: learn_rate must be a float or int greater than 0

    # Should create an instance of the MLP class
    # >>> mlp = MLP()
    # >>> mlp
    # MLP()

    # Should raise an error if an invalid input is passed to add_layer
    # >>> mlp = MLP()
    # >>> mlp.add_layer(np.array([]))
    # Traceback (most recent call last):
    # ...
    # ValueError: layer must be of at least length 1

    # Should add the input layer and a hidden layer
    # >>> mlp = MLP(hidden_layer_size=(4,2))
    # >>> mlp.add_layer(np.array([Perceptron() for _ in range(2)]))
    # >>> mlp.layers
    # array([array([Perceptron(), Perceptron()], dtype=object),
    #        array([[Perceptron(), Perceptron()],
    #               [Perceptron(), Perceptron()],
    #               [Perceptron(), Perceptron()],
    #               [Perceptron(), Perceptron()]], dtype=object)], dtype=object)


import unittest
from mlp import MLP


class TestMLP(unittest.TestCase):
    
    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            pass     
