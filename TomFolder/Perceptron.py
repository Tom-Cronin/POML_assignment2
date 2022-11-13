import random

import numpy as np

class Perceptron:

    def __init__(self, learning_iterations=500, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.learning_iterations = learning_iterations
        self.activation = self.step_function
        self.weights = None
        self.bias = None

    def convert_label_to_num(self, labels):
        number_label = []
        for label in labels:
            if 'no' in label.lower():
                number_label.append(0)
            elif 'yes' in label.lower():
                number_label.append(1)
        return number_label

    def fit(self, data, labels):
        converted_labels = self.convert_label_to_num(labels)
        number_exampls, number_features = data.shape
        self.weights = np.zeros(number_features)
        self.bias = 0

        for loop in range(self.learning_iterations):
            for index, x_i in enumerate(data):
                prediction = self.activation(np.dot(x_i, self.weights) + self.bias)

                update = self.learning_rate * (converted_labels[index] - prediction)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        return self.activation(np.dot(X, self.weights) + self.bias)

    def step_function(self, z):
        # if z > 0:
        #     return 1
        # return 0
        return np.where(z>0, 1, 0)




#%%

#%%
