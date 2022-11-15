import numpy as np

from Layer import Layer
from Utils import *
from Metrics import * 

class MLP():
    """ A multi-layer perceptron class (neural net)

    Args:
    learn_rate -- float: the learning rate value of the neural net
    n_iters -- int: the number of iterations that will be carried out
    hidden_layer_size -- tuple: the # of perceptrons / hidden layer, and # of hidden layers to be added
    """

    def __init__(self):
        self.layers = [
            Layer(9, 'relu'),
            Layer(2, 'softmax')
        ]
        pass
        
    def init_params(self):
        W1 = np.random.rand(9, 184)
        b1 = np.random.rand(184, 2)
        W2 = np.random.rand(2, 184)
        b2 = np.random.rand(2, 2)
        return W1, b1, W2, b2
    
    def forward_propegation(self, X):
        z1 = self.layers[0].predict(X)
        A1 = np.maximum(0, z1) # relu
        z2 = self.layers[0].predict(X)
        A2 = np.exp(A1) / np.sum(np.exp(A1)) 
        
        return z1, A1, z2, A2
    
    def one_hot(self, y):
        o_Y = np.zeros((y.size, y.max() + 1))
        o_Y[np.arange(y.size), y] = 1
        return o_Y.T
        
    
    def back_propegation(self, Z1, A1, Z2, A2, W2, Y, X):
        m = Y.size
        one_hot_y = self.one_hot(Y)
        dZ2 = A2 - one_hot_y
        dW2 = 1 / m * dZ2.dot(Z2.T)
        db2 = 1 / m * np.sum(dZ2, 2)
        dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, 2)
        return dW1, db1, dW2, db2
    
    def update_params(self, W1, b1, W2, b2, dW1, dbl, dW2, db2, a):
        W1 = W1 - a * dW1
        b1 = b1 - a * dbl
        W2 = W2 - a * dW2
        b2 = b2 - a * db2
        return W1, b1, W2, b2
        
    def grad_desc(self, X, Y, iters, a):
        W1, b1, W2, b2 = self.init_params()
        for i in range(iters):
            Z1, A1, Z2, A2 = self.forward_propegation(X)
            dW1, db1, dW2, db2 = self.back_propegation(Z1, A1, Z2, A2, W2, Y, X)
            W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, a)
            if i % 50 == 0:
                print('iter:', i)
                print('Acc: ', np.sum(np.argmax(A2, 0) == Y))
        return W1, b1, W2, b2
            
if __name__=='__main__':
    
    wildfires = read_data_return_dataframe("wildfires.txt")
    # Copy to be used for the rest of the assignment
    wildfires_copy = wildfires.copy()
    # wildfires_copy = convert_label(wildfires,'fire',['no', 'yes'],[0, 1])


    features = ['year', 'temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'day', 'month', 'wind_speed']
    X_train, X_test, y_train, y_test = split_df_to_train_test_dfs(wildfires_copy, test_set_size=.1,
                                                        random_state=42)
    # X_train = X_train[features].values  # returns a numpy NdArray of the features
    # # X_test = X_test[features].values  # returns a numpy NdArray of the features
    # X_train = Normalize(X_train, features)
    # X_test = Normalize(X_test, features)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).flatten()
    y_train = np.asarray([1 if 'yes' in y else 0 for y in y_train])
    mlp = MLP()
    W1, bl, W2, b2 = mlp.grad_desc(X_train, y_train, 500, 0.1)
    
    

    # def impl(self):
    #     [l1, l2, l3] = self.layers
        
        
    
#     def add_layer(self, layer):
#         self.layers.append(layer)

#     def forward_propegation(self, X, W, B):
#         for layer in self.layers:
            

#         pass
#         # for h in layer[0]:

#         # for h in layer:
#         #     for h_i in h:
#         #         print(h_i)

#     def fit(self, X, y):
#         # for e/a perceptronfo in each layer call fit
#         # need to forward_propegate the weights and bias from the perceptron, to the next... I think 
#         for h in self.layers[0]:
#             h.fit(X,y)
#             W = h.weights
#             B = h.bias
#             self._feed_forward(self.layers[1], X=X, W=W, B=B, y=y)


#     def evaluate(self, X, y):
#         pass

#     def predict(self):
#         pass

#     def fit_predict(self):
#         pass

#     def __repr__(self):
#         return f"{type(self).__name__}()"

#     def _validate_input_params(self, learn_rate, n_iters, hidden_layer_size):
#         if not isinstance(n_iters, int) or n_iters < 1:
#             raise ValueError("n_iters must be an integer and a natural number")
#         if not isinstance(learn_rate, (int, float)) or learn_rate <= 0:
#             raise ValueError("learn_rate must be a float or int greater than 0")
#         n, m = hidden_layer_size
#         if not isinstance(n, int) or n < 1:
#             raise ValueError("hidden_layer_size must contain natural numbers of type int")
#         if not isinstance(m, int) or m < 1:
#             raise ValueError("hidden_layer_size must contain natural numbers of type int")

# # # mlp.add_layer(np.array([]))

# # print(mlp.layers)
# # mlp.add_layer([[1,2,3]])
# # mlp.add_layer([[4,5,3]])
# # mlp.add_layer([[6,7,3]])