from activations import *

class Neuron():
    def __init__(self, weights, bias, activation):
        """
        Constructor
        
        X : array of inputs
        weigths : array of weights
        bias : constant
        activation : the activation function for the neuron
        
        """
        self.weights = weights
        self.bias = bias
        self.activation = activation
    
    def feedforward(self, X):
        if self.activation is "sig":
            return sigmoid(np.dot(X, self.weights) + self.bias)