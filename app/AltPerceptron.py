import pandas as pd
import numpy as np
# Daniel Verdejo - a simple implementation of the perceptron class UNUSED
class Perceptron:
    """The perceptron class"""

    def __init__(self, learn_rate=0.001, n_iters=500, weights=None, bias=None):
        self._validate_input_params(learn_rate, n_iters)
        self.learn_rate = learn_rate
        self.n = n_iters
        self.activation_fn = (lambda x: np.where(x >= 0, 1, 0)) # Heaviside step function
        self.W = weights
        self.b = bias

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __str__(self):
        return f'{type(self).__name__} - weights:{self.W}, bias:{self.b}'

    def _validate_input_params(self, learn_rate, n_iters):
        if not isinstance(n_iters, int) or n_iters < 1:
            raise ValueError("n_iters must be an integer and a natural number")
        if not isinstance(learn_rate, (int, float)) or learn_rate <= 0:
            raise ValueError("learn_rate must be a float or int greater than 0")


    def fit(self, X, y):
        if(isinstance(X, pd.DataFrame)):
            X = X.to_numpy()

        if self.W is None:
            self.W = np.random.rand(X.shape[1])
            self.b = 0

        for _ in range(self.n):
            for i, x_i in enumerate(X):
                z = np.dot(x_i, self.W) + self.b
                update = self.learn_rate * (y[i] - self.activation_fn(z))

                self.W = self.W + (update * x_i)
                self.b += update

        return self

    def predict(self, X):
        if(isinstance(X, pd.DataFrame)):
            X = X.to_numpy()
        z = np.dot(X, self.W) + self.b
        return self.activation_fn(z)