from activations import *
import numpy as np

class Layer():
    def __init__(self, input_dim, output_dim, activation):
        """
        Constructor
        
        input_dim : number of inputs for the layer (optional)
        output_dim : number of outputs for the layer, essentially the number of neurons in the layer
        
        """
        self.activations = np.array([])
        self.activation = activation
        self.weights = np.random.rand(output_dim, input_dim)
        self.biases = np.random.rand(output_dim)
        
    def feedforward(self, input_layer):
        input_activations = input_layer.activations
        
        activations = np.add(np.dot(self.weights, input_activations), self.biases)
        
        if self.activation is 'sig':
            self.activations = np.vectorize(sigmoid)(activations)
        
        
L = Layer(4, 2, 'sig')
input_L = Layer(8, 4, 'sig')
input_L.activations = np.random.rand(4)

L.feedforward(input_L)

print(L.activations)