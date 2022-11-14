import random

import numpy as np

class Perceptron():
    #
    # def __init__(self, learning_iterations=500, learning_rate=0.1, bias=None, passthrough=False):
    #     self.learning_rate = learning_rate
    #     self.learning_iterations = learning_iterations
    #     self.activation = self.step_function
    #     self.weights = None
    #     self.bias_1 = bias
    #     self.passthrough = passthrough
    #
    # def convert_label_to_num(self, labels):
    #     number_label = []
    #     for label in labels:
    #         if 'no' in label.lower():
    #             number_label.append(0)
    #         elif 'yes' in label.lower():
    #             number_label.append(1)
    #     return number_label
    #
    # def fit(self, data, labels):
    #
    #
    #     converted_labels = self.convert_label_to_num(labels)
    #     number_exampls, number_features = data.shape
    #     self.weights = np.zeros(number_features)
    #
    #     for loop in range(self.learning_iterations):
    #         for index, x_i in enumerate(data):
    #             prediction = self.activation(np.dot(x_i, self.weights))
    #
    #             update = self.learning_rate * (converted_labels[index] - prediction)
    #             self.weights += update * x_i
    #
    # def predict(self, X):
    #     return self.activation(np.dot(X, self.weights))
    #
    # def fit_predict(self, Data, y_labels):
    #     if self.passthrough:
    #         return Data
    #     else:
    #         self.fit(Data, y_labels)
    #         return self.predict(Data)
    # def step_function(self, z):
    #     if self.bias_1:
    #         return 1
    #     # if z > 0:
    #     #     return 1
    #     # return 0
    #     return np.where(z>0, 1, 0)

    def __init__(self, input_size, bias=0):
        self.weights = np.random.uniform(low=-1, high=1, size =(input_size,)) # initalises weights randomly
        self.bias = bias
        pass

    def fit(self, inputs):  # takes an inpud 1darray
        pass

    def predict(self, inputs):
        return np.dot(self.weights, inputs) + self.bias
        pass

if __name__ == '__main__':
    inputs = [[1, 2,3.2, 5],
              [2,5,2, 5],
              [1,1,3,5 ]]

    weights = [[3,2,1, 6],
               [3,5,6, 6]
               ,[7,9,4, 6]]
    input_array = np.array(inputs)
    print(np.dot(np.array(weights).T, input_array))
    # bias = 3
    # test = 1 + 4 + (3.2 * 5) + bias
    #
    # perceptron= Perceptron(len(inputs), bias)
    # print(perceptron.predict(inputs))