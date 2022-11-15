import numpy as np

from ThresholdLogicUnit import ThresholdLogicUnit as TLU
from Utils import *
from Metrics import *

class Layer:
    
    def __init__(self, output_size, activation, input_size):
        self._validate_input_params(output_size, input_size)
        self.size = output_size
        self.input_size = input_size
        self.N = [TLU(learning_rate=0.001, activation_function=activation) for _ in range(output_size)]
        self.init_weights(self.input_size)
        
    
    def _validate_input_params(self, output_size, input_size):
        if not isinstance(output_size, int) or output_size < 1:
            raise ValueError("size must be an integer and a natural number")
        if not isinstance(input_size, int) or input_size < 1:
            raise ValueError("size must be an integer and a natural number")
    
    def fit(self, X, y=None):
        [ n.fit(X, y) for n in self.N ]
    
    def predict(self, X):
        pred = [n.predict(X) for n in self.N]
        return np.asarray(pred)
    
    def init_weights(self, X):
        [ n.initialise_weights(np.zeros(self.input_size)) for n in self.N ]
    
    def update_weights(self, W):
        for n in self.N:
            for w in n.weights:
                w += W

                



        
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

    l1 = Layer(32, "relu", input_size=9)
    pred = l1.predict(X_train)

    l2 = Layer(1, "sigmoid", input_size=32)
    # l2.init_weights(X_train)
    # # l2.fit(predictions, y_train)
    prediction = l2.predict(pred)
    print(prediction)
    
    
