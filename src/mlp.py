class MLP():
    """ A multi-layer perceptron class

    Args:
    learn_rate -- float: the learning rate value of the neural net
    n_iters -- int: the number of iterations that will be carried out
    hidden_layer_size -- tuple: the # of perceptrons / hidden layer, and # of hidden layers to be added
    """

    def __init__(self, learn_rate=0.001, n_iters=100, hidden_layer_size=(50,1)):
        self._validate_input_params(learn_rate, n_iters, hidden_layer_size)
        self.learn_rate = learn_rate
        self.n_iters = n_iters
        self.hidden_layer_size = hidden_layer_size

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
        if len(layer) == 0:
            raise ValueError("layer must be of at least length 1")
        if hasattr(self, 'layers'):
            ls = self.layers.tolist()
            ls.append(layer)
            self.layers = np.array(ls, dtype=object)
        else:
            n, m = self.hidden_layer_size
            ls = [[Perceptron() for _ in range(n)] for _ in range(m) ]
            ls.insert(0, layer.tolist())
            self.layers = np.array(ls)

    def fit(self, X, y):
        # for e/a perceptron in each layer call fit
        # need to forward_propegate the weights and bias from the perceptron, to the next... I think 

        for P in self.layers:
            for p in P:
                p.fit(X, y)

    def evaluate(self, X, y):
        pass

    def predict(self):
        pass

    def fit_predict(self):
        pass

df = read_data_return_dataframe("./wildfires.txt")
df_train, df_test = split_df_to_train_test_dfs(df)
y_train, X_train = split_df_labels_attributes(df_train)
y_train = normalise_outputs(y_train)
    
mlp = MLP(learn_rate=.02, n_iters= 500)
mlp.add_layer(np.array([Perceptron() for _ in range(4)]))
mlp.fit(X_train, y_train)
# # mlp.add_layer(np.array([]))

# print(mlp.layers)
# mlp.add_layer([[1,2,3]])
# mlp.add_layer([[4,5,3]])
# mlp.add_layer([[6,7,3]])