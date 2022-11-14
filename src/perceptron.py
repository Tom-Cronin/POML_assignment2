import pandas as pd
import numpy as np

class Perceptron:
    """The perceptron class"""

    def __init__(self, learn_rate=0.001, n_iters=500, weights=None, bias=None):
        self._validate_input_params(learn_rate, n_iters)
        self.learn_rate = learn_rate
        self.n_iters = n_iters
        self.activation_fn = (lambda x: np.where(x >= 0, 1, 0)) # Heaviside step function
        self.weights = weights
        self.bias = bias

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __str__(self):
        return f'{type(self).__name__} - weights:{self.weights}, bias:{self.bias}'

    def _validate_input_params(self, learn_rate, n_iters):
        if not isinstance(n_iters, int) or n_iters < 1:
            raise ValueError("n_iters must be an integer and a natural number")
        if not isinstance(learn_rate, (int, float)) or learn_rate <= 0:
            raise ValueError("learn_rate must be a float or int greater than 0")


    def fit(self, X, y=None):

        if(isinstance(X, pd.DataFrame)):
            X = X.to_numpy()
        
        if self.weights is None:
            self.weights = np.random.rand(X.shape[1])
            self.bias = 0

        for _ in range(self.n_iters):
            for i, x_i in enumerate(X):
                z = np.dot(x_i, self.weights) + self.bias
                update = self.learn_rate * (y[i] - self.activation_fn(z))

                self.weights = self.weights + (update * x_i)
                self.bias += update
                
        return self

    def predict(self, X):
        if(isinstance(X, pd.DataFrame)):
            X = X.to_numpy()
        z = np.dot(X, self.weights) + self.bias
        return self.activation_fn(z)
