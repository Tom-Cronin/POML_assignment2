import numpy as np

import src.perceptron as p

class Layer:
    
    def __init__(self, size, lr=0.001, n=500, W=None, B=None):
        self.size = size
        self.Xs = [p.Perceptron(lr, n, W, B) for _ in range(size)]

    def __str__(self):
        return 'layer'