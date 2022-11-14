# import math

# from Perceptron import Perceptron

# import numpy as np

# class Layer():

#     def __init__(self, number_neurons, learning_rate, learning_iters, passthrough=False, bias_neuron=False):
#         self.number_neurons = number_neurons
#         self.learning_rate = learning_rate
#         self.learning_iters = learning_iters
#         self.passthrough = passthrough
#         self.bias_neuron = bias_neuron
#         self.perceptrons = []

#     def createPerceptrons(self):

#         if self.bias_neuron:
#             self.perceptrons.append(Perceptron(learning_rate=self.learning_rate,
#                                                learning_iterations=self.learning_iters,
#                                                bias=1,
#                                                passthrough=self.passthrough))
#         for i in range(self.number_neurons):
#             self.perceptrons.append(Perceptron(learning_rate=self.learning_rate,
#                                                learning_iterations=self.learning_iters,
#                                                passthrough=self.passthrough))

#     def predict(self, X):
#         for neuron in self.perceptrons:
#             return neuron.predict(X)

#     def fit(self, Data, Labels):
#         output = []
#         if self.passthrough:
#             for n in self.perceptrons:
#                 if n.bias_1:
#                     output.append(1)
#                 else:
#                     for d in Data:
#                         output.append(Data)

#         for neuron in self.perceptrons:
#             output.append(neuron.fit_predict(Data, Labels))
#         return output


# class DenseLayer():
#     def __init__(self, n_nuerons, n_inputs):
#         print(n_nuerons, n_inputs)
#         self.weights = 0.5 * np.random.randn(n_inputs, n_nuerons)
#         self.biases = np.zeros(n_nuerons)
#     def forward(self, inputs):
#         self.output = np.dot(inputs, self.weights) + self.biases


# class Relu:
#     def forward(self, inputs):
#         self.output = np.maximum(0, inputs)

# class soft_max():
#     def forward(self, inputs):
#         exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
#         self.output = exp / np.sum(exp, axis=1, keepdims=True)
#         self.preds = []
#         for pred in self.output:
#             self.preds.append(max(pred))
# class loss_function():

#     def conver_labels(self, labels):
#         converted = []
#         for label in labels:
#             if 'no' in label.lower():
#                 converted.append(0)
#             else:
#                 converted.append(1)
#         return converted
#     def loss(self, preds, labels):
#         lost =0
#         labels = self.conver_labels(labels)
#         for index in range(len(labels)):
#             lost += math.log(preds[index]) * labels[0]


#         return lost

