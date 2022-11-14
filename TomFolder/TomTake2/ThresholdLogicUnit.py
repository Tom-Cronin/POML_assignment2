import random
import numpy as np
import data_functions
from data_functions import Normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

class ThresholdLogicUnit:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weights = None

    def __inialiseWeights(self, x):
        w = []
        for _ in range(len(x)):
            w.append(random.uniform(-.5,.5))
        self.weights = np.asarray(w)

    def sgn_step_fn(self, sum_weights):
        if sum_weights >0:
            return 1
        return 0

    def fit(self, X, y, learning_iterations=200):
        if self.weights is None:
            self.__inialiseWeights(X[0])  # sets the weights to the amount of inputs

        for _ in range(learning_iterations):
            for (data_vector, label) in zip(X, y):
                prediction = self.sgn_step_fn(np.dot(data_vector, self.weights))
                if prediction != label:
                    error = prediction - label
                    self.weights += -self.learning_rate * error * data_vector

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.sgn_step_fn(np.dot(x, self.weights)))
        return prediction





















def accuracy(predictions, labels):
    right = 0
    wrong = 0
    for index in range(len(predictions)):
        if predictions[index] == labels.values[index]:
            right +=1
        else:
            wrong +=1
    return right/(wrong+right)


#  one pass = forward + backwards
if __name__=='__main__':
    wildfires = data_functions.read_data_return_dataframe("../../wildfires.txt")
    # Copy to be used for the rest of the assignment
    wildfires_copy = wildfires.copy()
    wildfires_copy = data_functions.convert_label(wildfires,
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

    perceptron = ThresholdLogicUnit(learning_rate=0.1)
    perceptron.fit(X_train,y_train, learning_iterations=2)

    predictions = perceptron.predict(np.asarray(X_test))
    print(np.asarray(predictions))
    print(y_test)
    print(accuracy(predictions, y_test))








