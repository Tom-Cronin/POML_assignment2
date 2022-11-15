import numpy as np

from Layer import Layer
from Utils import *
from Metrics import * 

# Daniel Verdejo - MLP class
class MLP():
    """ A multi-layer perceptron class (neural net)"""

    def __init__(self):
        self.layers = []
        self.weights = []
    
    def add_layer(self, output_size, activation, input_size):
        """ add_layer
        args:
        output_size -- the number of outputs from the layer
        activation -- the activation function to use "relu", "sigmoid", "heaviside"
        input_size -- the number of inputs for the layer
        """
        self.layers.append(Layer(output_size, activation, input_size))
        self.weights.append((output_size, input_size))
    
    def forward_propegation(self, X):
        nextZ =  self.layers[0].predict(X) # for the first layer feed raw input
        for i in range(1, len(self.layers)): 
            nextZ = self.layers[i].predict(nextZ) # for every other layer feed the output of every neuron to the next layer of neurons

        return nextZ # return the outputs layer
        
    
    def back_propegation(self, Z, X, Y):
        sig_deriv = (lambda x: x * (1 - x))
        e = Z - Y.T # get the error of our output
        delta = e * sig_deriv(Z) # using the derivative of sigmoid to get the difference
        update_0 = []
        for w in range(1,len(self.weights)): # for every weight in our weight we gather the err and delta and update the weights accordingly
            z_err = delta.dot(w) # the error of the layer based off the final output delta
            z_delta = z_err * sig_deriv(Z) # the diff between the layer output and final output
            for d in z_delta:
                if (d.shape == X.shape):
                    update_0 += X.T.dot(d) # the update should be the dot product of our transposed X and delta of layer deltas
            self.layers[0].update_weights(update_0) # update the weights of our input to hidden layer
            update_l_n =[]
            for d in delta:
                if d.shape == Z.T.shape:
                    update_l_n += Z.T.dot(d)
            self.layers[w].update_weights(update_l_n) # update all other layers weights
    
    def train(self, X, Y, iters, show_metrics=False):
        for i in range(iters):
            Z = self.forward_propegation(X) # feed forward the data
            self.back_propegation(Z, Y, X) # propegate back the error and delta to update the weights on each neuron of layer n
            if show_metrics and i % 25 == 0:
                print(f'iteration: {i} / {iters} ============= Acc: {accuracy(Z[0], Y)}')# check our gradient descent
        print(f'Prediction: {Z}')
        return Z # finally return our prediction
    
    def predict(self, X):
        p = self.forward_propegation(X)
        return p

if __name__=='__main__':
    wildfires = read_data_return_dataframe("wildfires.txt")
    # Copy to be used for the rest of the assignment
    wildfires_copy = wildfires.copy()
    # wildfires_copy = convert_label(wildfires,'fire',['no', 'yes'],[0, 1])

    features = ['year', 'temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'day', 'month', 'wind_speed']
    X_train, X_test, y_train, y_test = split_df_to_train_test_dfs(wildfires_copy, test_set_size=.2, random_state=42)
    X_train = X_train[features].values  # returns a numpy NdArray of the features
    X_test = X_test[features].values  # returns a numpy NdArray of the features
    X_train = Normalize(X_train, features)
    X_test = Normalize(X_test, features)

    X_train = np.asarray(X_train)[0:40]
    y_train = np.asarray(y_train).flatten()
    y_test = np.asarray(y_test).flatten()

    y_train = np.asarray([1 if 'yes' in y else 0 for y in y_train])[0:40]
    X_test = np.asarray(X_test)[0:40]
    y_test = np.asarray([1 if 'yes' in y else 0 for y in y_test])[0:40]


    m, n = X_train.shape
    mlp = MLP()
    mlp.add_layer(output_size = m, activation='relu', input_size=n) # Add a layer of 32 inputs
    mlp.add_layer(output_size = 1, activation='sigmoid', input_size= m)
    mlp.train(X=X_train, Y=y_train, iters= 1000, show_metrics= True)
    
    p = mlp.predict(X_test)

    print(p)
    