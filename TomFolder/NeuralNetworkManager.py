from Layers import Layer
import numpy as np


class NeuralNetwork:

    def __init__(self, learning_rate, learning_iterations):
        self.layers = []
        self.learning_rate = learning_rate
        self.learning_iterations = learning_iterations


    def add(self, number_of_perceptrons, passthrough=False, bias_neuron=False):
        self.layers.append(Layer(number_of_perceptrons, self.learning_rate, self.learning_iterations, passthrough,bias_neuron))

    def fit(self, Data, labels):
        layer_output = Data
        for layer in range(len(self.layers)):
            layer_output = np.asarray(self.layers[layer].fit(layer_output, labels))

    def predict(self, X):
        for layer in self.layers:
            layer.predict(X)




