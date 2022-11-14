from Layers import DenseLayer, Layer, Relu, soft_max, loss_function
import numpy as np
import data_functions
from data_functions import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

X = [[1,2,3,2.5],
     [2.0, 5.0, -1, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


weights = []
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

    def add_dense(self, input_shape, number_of_nuerons):
        self.layers.append(DenseLayer(number_of_nuerons, input_shape))





if __name__=='__main__':
    activation = Relu()

    wildfires = data_functions.read_data_return_dataframe("../wildfires.txt")
    # Copy to be used for the rest of the assignment
    wildfires_copy = wildfires.copy()
    wildfires_copy = data_functions.convert_label(wildfires,
                                                  'fire',
                                                  ['no', 'yes'],
                                                  ['NO', 'YES'])
    wildfires_labels = wildfires_copy.copy()['fire']
    wildfires_copy.drop('fire', axis=1, inplace=True)

    features = ['year', 'temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'day', 'month', 'wind_speed']
    X_train, X_test, y_train, y_test = train_test_split(wildfires_copy, wildfires_labels, test_size=0.3,
                                                        random_state=42)
    X_train = X_train[features].values  # returns a numpy NdArray of the features
    X_test = X_test[features].values  # returns a numpy NdArray of the features
    X_train = Normalize(X_train, features)
    X_test = Normalize(X_test, features)


    dnn = NeuralNetwork(1,1)
    dnn.add_dense(9, 5)
    dnn.add_dense(5,2)
    relu_act = Relu()
    sm = soft_max()
    print(X_train.shape)
    lf = loss_function()
    dnn.layers[0].forward(X_train)
    relu_act.forward(dnn.layers[0].output)
    dnn.layers[1].forward(relu_act.output)
    sm.forward(dnn.layers[1].output)
    # print(sm.preds)
    print(sm.preds[0])
    print(lf.loss(sm.preds, y_train))
