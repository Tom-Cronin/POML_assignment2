import numpy as np

from ThresholdLogicUnit import ThresholdLogicUnit as TLU
from Utils import *
from Metrics import *

class Layer:
    
    def __init__(self, size, activation, lr=0.001):
        self._validate_input_params(size)
        self.size = size
        self.N = [TLU(learning_rate=lr, activation_function=activation) for _ in range(size)]
        
    
    def _validate_input_params(self, size):
        if not isinstance(size, int) or size < 1:
            raise ValueError("size must be an integer and a natural number")
    
    def fit(self, X, y=None):
        [ n.fit(X, y) for n in self.N ]
    
    def predict(self, X):
        return [ n.predict(X) for n in self.N ]
    
    def init_weights(self, X):
        print(X.shape)
        [ n.initialise_weights(X[0]) for n in self.N ]


        
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

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).flatten()
    y_train = np.asarray([1 if 'yes' in y else 0 for y in y_train])

    
    l1 = Layer(9, "relu")
    l1.init_weights(X_train)
   

    predictions = np.asarray(l1.predict(X_test))

    l2 = Layer(2, "softmax")
    l2.init_weights(predictions)
    l2.fit(predictions, y_train)
    
    prediction = l2.predict(X_test.T)
    print(prediction)



    # # print("Test", perceptron.sigmoid(0))
    # print("Confusin Matrix [TP, FP][TN, FN]")

    # # cf_m = confusion_matrix(predictions, y_test.values)
    # # print("True Positives = ", cf_m[0][0])
    # # print("False Positives = ", cf_m[0][1])
    # # print("True Negative = ", cf_m[1][1])
    # # print("False Negative = ", cf_m[1][0])

    # # print("Accuracy: ", accuracy(predictions, y_test.values))
    # # print("Precision: ", precision(predictions, y_test.values))
    # # print("Recall: ", recall(predictions, y_test.values))
    # # print("F1 Score: ", f1_score(predictions, y_test.values))
