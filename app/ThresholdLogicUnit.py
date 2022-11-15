import random
import numpy as np
from Utils import *
from sklearn.model_selection import train_test_split
from math import e, pow
from Metrics import *
# Tom Cronin, Creating the Perceptron for a single layer


class ThresholdLogicUnit:

    def __init__(self, learning_rate, activation_function='heaviside'):
        self.learning_rate = learning_rate  # specifies the learning rate for the TLU
        self.weights = None                 # none for first time activation can be passed pre clalced biases
        self.activation_function = activation_function # specifies activation function 'Relu', 'Sigmoid', 'Heaviside'

    def initialise_weights(self, x):
        w = []  # empty weights array
        for _ in range(len(x)):  # loops through inputs
            w.append(random.uniform(-.5,.5))  # initialises weights randomly between -.5 and +.5
        self.weights = np.asarray(w)  # converts and sets weights to numpy array at the instance level

    def heaviside(self, sum_weights):  # base function
        if sum_weights >= 0:  # returns 1 if score is positive
            return 1
        return 0  # returns 0 if score is negative

    def relu(self, sum_weights):  # activation function for layers
        if sum_weights >= 0:  # returns the value if positive
            return sum_weights
        return 0  # returns 0 if value is negative

    def sigmoid(self, sum_weights):  # sigmoid activation for output
        results = 1 / (1 + (pow(e,-sum_weights)))  # calculates the sigmoid result
        if results >= 0.5:  # ie if sigmoid  a 0 sum_weight value return 1 or higher return positive
            return 1
        return 0

    def activation_func(self, sum_weights): # switches based on function to use
        if self.activation_function.lower() == "relu":
            return self.relu(sum_weights)
        elif self.activation_function.lower() == "sigmoid":
            return self.sigmoid(sum_weights)
        else:
            return self.heaviside(sum_weights) # default is heaviside

    def fit(self, X, y=None, learning_iterations=200):  # sets the default iterations to be 200
        if self.weights is None:
            self.initialise_weights(X[0])  # sets the weights to the amount of inputs
        for _ in range(learning_iterations):  # loops through all the iterations
            for (data_vector, label) in zip(X, y):
                prediction = self.activation_func(np.dot(data_vector.T, self.weights))  # runs activation
                if prediction != label:  # if it gets the wrong prediction update the weights
                    error = prediction - label
                    self.weights = self.weights - (self.learning_rate * error * data_vector)

    def predict(self, X):  # predicts each sample
        prediction = []
        for x in X:
            prediction.append(self.activation_func(np.dot(x.T, self.weights)))
        return prediction
    def __repr__(self):
        return f"{type(self).__name__}()"

    def __str__(self):
        return f'{type(self).__name__} - weights:{self.weights}'

    def _validate_input_params(self, learn_rate, n_iters):
        if not isinstance(n_iters, int) or n_iters < 1:
            raise ValueError("n_iters must be an integer and a natural number")
        if not isinstance(learn_rate, (int, float)) or learn_rate <= 0:
            raise ValueError("learn_rate must be a float or int greater than 0")

#  one pass = forward + backwards
if __name__=='__main__':
    wildfires = read_data_return_dataframe("../wildfires.txt")
    # Copy to be used for the rest of the assignment
    wildfires_copy = wildfires.copy()
    wildfires_copy = convert_label(wildfires,
                                                  'fire',
                                                  ['no', 'yes'],
                                                  ['NO', 'YES'])
    wildfires_labels = wildfires_copy.copy()['fire']
    wildfires_copy.drop('fire', axis=1, inplace=True)

    ndarray = wildfires_labels.copy()
    for index in range(len(ndarray)):
        if 'no' in ndarray[index].lower():
            ndarray[index] = 0
        elif 'yes' in ndarray[index].lower():
            ndarray[index] = 1
    wildfires_labels = ndarray

    features = ['year', 'temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'day', 'month', 'wind_speed']
    X_train, X_test, y_train, y_test = train_test_split(wildfires_copy, wildfires_labels, test_size=0.1,
                                                        random_state=42)
    X_train = X_train[features].values  # returns a numpy NdArray of the features
    X_test = X_test[features].values  # returns a numpy NdArray of the features
    X_train = Normalize(X_train, features)
    X_test = Normalize(X_test, features)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    perceptron = ThresholdLogicUnit(learning_rate=0.1, activation_function='sigmoid')
    perceptron.fit(X_train,y_train, learning_iterations=200)

    predictions = perceptron.predict(np.asarray(X_test))
    print(predictions)


    print("Test", perceptron.sigmoid(0))
    print("Confusin Matrix [TP, FP][TN, FN]")

    cf_m = confusion_matrix(predictions, y_test.values)
    print("True Positives = ", cf_m[0][0])
    print("False Positives = ", cf_m[0][1])
    print("True Negative = ", cf_m[1][1])
    print("False Negative = ", cf_m[1][0])

    print("Accuracy: ", accuracy(predictions, y_test.values))
    print("Precision: ", precision(predictions, y_test.values))
    print("Recall: ", recall(predictions, y_test.values))
    print("F1 Score: ", f1_score(predictions, y_test.values))

#%%

#%%

#%%
