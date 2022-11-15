import numpy as np

from Layer import Layer
from Utils import *
from Metrics import * 

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
        nextZ =  self.layers[0].predict(X)
        for i in range(1, len(self.layers)):
            nextZ = self.layers[i].predict(nextZ)

        return nextZ # return the outputs of each layer
        
    
    def back_propegation(self, Z, X, Y):
        sig_deriv = (lambda x: x * (1 - x))
        e = Z - Y.T # get the error of our output
        delta = e * sig_deriv(Z) # using the derivative of sigmoid to get the difference
        update = []
        for w in range(1,len(self.weights)): # for every weight in our weight we gather the err and delta 
            z_err = delta.dot(w)
            z_delta = z_err * sig_deriv(Z)
            for d in z_delta:
                if (d.shape == X.shape):
                    update += X.T.dot(d)
            self.layers[0].update_weights(update)
            x =[]
            for d in delta:
                if d.shape == Z.T.shape:
                    x += Z.T.dot(d)
            self.layers[w].update_weights(x) 

    
        
    def gradient_desc(self, X, Y, iters, a):
        for i in range(iters):
            Z = self.forward_propegation(X)
            self.back_propegation(Z, Y, X)
            if i % 50 == 0:
                print('iter:', i)
                print('Acc: ', np.sum(np.argmax(Z, 0) == Y))
        return Z
            
if __name__=='__main__':
    wildfires = read_data_return_dataframe("wildfires.txt")
    # Copy to be used for the rest of the assignment
    wildfires_copy = wildfires.copy()
    # wildfires_copy = convert_label(wildfires,'fire',['no', 'yes'],[0, 1])


    features = ['year', 'temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'day', 'month', 'wind_speed']
    X_train, X_test, y_train, y_test = split_df_to_train_test_dfs(wildfires_copy, test_set_size=.1,
                                                        random_state=42)
    X_train = X_train[features].values  # returns a numpy NdArray of the features
    X_test = X_test[features].values  # returns a numpy NdArray of the features
    X_train = Normalize(X_train, features)
    X_test = Normalize(X_test, features)

    X_train = np.asarray(X_train)[0:32]
    y_train = np.asarray(y_train).flatten()
    y_train = np.asarray([1 if 'yes' in y else 0 for y in y_train])[0:32]


    m, n = X_train.shape
    print(m, n)
    mlp = MLP()
    mlp.add_layer(output_size = m, activation='relu', input_size=n) # Add a layer of 32 inputs
    mlp.add_layer(output_size = 1, activation='sigmoid', input_size= m)
    Z = mlp.gradient_desc(X_train, y_train, 500, 0.1)
    
    print(Z)
    