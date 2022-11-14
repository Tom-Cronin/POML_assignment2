import numpy as np

# import src.layer as l

class MLP():
    """ A multi-layer perceptron class (neural net)

    Args:
    learn_rate -- float: the learning rate value of the neural net
    n_iters -- int: the number of iterations that will be carried out
    hidden_layer_size -- tuple: the # of perceptrons / hidden layer, and # of hidden layers to be added
    """

    def __init__(self, learn_rate=0.001, n_iters=100, hidden_layer_size=(25,1)):
        self._validate_input_params(learn_rate, n_iters, hidden_layer_size)
        self.learn_rate = learn_rate
        self.n_iters = n_iters
        self.hidden_layer_size = hidden_layer_size
        self.layers = []

    def __repr__(self):
        return f"{type(self).__name__}()"

    def _validate_input_params(self, learn_rate, n_iters, hidden_layer_size):
        if not isinstance(n_iters, int) or n_iters < 1:
            raise ValueError("n_iters must be an integer and a natural number")
        if not isinstance(learn_rate, (int, float)) or learn_rate <= 0:
            raise ValueError("learn_rate must be a float or int greater than 0")
        n, m = hidden_layer_size
        if not isinstance(n, int) or n < 1:
            raise ValueError("hidden_layer_size must contain natural numbers of type int")
        if not isinstance(m, int) or m < 1:
            raise ValueError("hidden_layer_size must contain natural numbers of type int")
    
    def add_layer(self, layer):
        # if type(layer) != l.Layer:
        #     raise TypeError("Must be of type Layer")
        
        self.layers.append(layer)

    def _feed_forward(self, layer, X, W, B, y, Z=None):
        for h in layer:
            h.weights, h.bias = W, B
            print(h)            
            h.fit(X, y)

        pass
        # for h in layer[0]:

        # for h in layer:
        #     for h_i in h:
        #         print(h_i)

    def fit(self, X, y):
        # for e/a perceptronfo in each layer call fit
        # need to forward_propegate the weights and bias from the perceptron, to the next... I think 
        for h in self.layers[0]:
            h.fit(X,y)
            W = h.weights
            B = h.bias
            self._feed_forward(self.layers[1], X=X, W=W, B=B, y=y)


    def evaluate(self, X, y):
        pass

    def predict(self):
        pass

    def fit_predict(self):
        pass

    

# # mlp.add_layer(np.array([]))

# print(mlp.layers)
# mlp.add_layer([[1,2,3]])
# mlp.add_layer([[4,5,3]])
# mlp.add_layer([[6,7,3]])