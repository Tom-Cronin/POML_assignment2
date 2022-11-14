from Perceptron import Perceptron

class Layer():

    def __init__(self, number_neurons, learning_rate, learning_iters, passthrough=False, bias_neuron=False):
        self.number_neurons = number_neurons
        self.learning_rate = learning_rate
        self.learning_iters = learning_iters
        self.passthrough = passthrough
        self.bias_neuron = bias_neuron
        self.perceptrons = []

    def createPerceptrons(self):

        if self.bias_neuron:
            self.perceptrons.append(Perceptron(learning_rate=self.learning_rate,
                                               learning_iterations=self.learning_iters,
                                               bias=1,
                                               passthrough=self.passthrough))
        for i in range(self.number_neurons):
            self.perceptrons.append(Perceptron(learning_rate=self.learning_rate,
                                               learning_iterations=self.learning_iters,
                                               passthrough=self.passthrough))

    def predict(self, X):
        for neuron in self.perceptrons:
            return neuron.predict(X)

    def fit(self, Data, Labels):
        output = []
        if self.passthrough:
            for n in self.perceptrons:
                if n.bias_1:
                    output.append(1)
                else:
                    for d in Data:
                        output.append(Data)

        for neuron in self.perceptrons:
            output.append(neuron.fit_predict(Data, Labels))
        return output


