from activations import *
import numpy as np

class Layer():
    def __init__(self, output_dim, activation="sig"):
        """
        Constructor
        
        input_dim : number of inputs for the layer (optional)
        output_dim : number of outputs for the layer, essentially the number of neurons in the layer
        
        """
        self.input_dim = 0
        self.output_dim = output_dim
        self.activations = np.array([])
        self.activation = activation
        self.weights = np.random.randn(self.output_dim, self.input_dim)/np.sqrt(self.input_dim + self.output_dim)
        self.biases = np.random.randn(output_dim)/np.sqrt(self.input_dim + self.output_dim)
        
        
    def __str__(self):
        return f"Layer -> input_dim : {self.input_dim}, output_dim : {self.output_dim}"
    
    
    def update_weights_biases(self, delta_weights, delta_biases, eta):
        self.weights = self.weights - eta * delta_weights
        self.biases = self.biases - eta * delta_biases
        
        
    def feedforward(self, input_layer):
        input_activations = input_layer.activations
        
        dot_product = np.dot(self.weights, input_activations)
        activations = np.add(dot_product, self.biases)
        # print(activations)
        
        if self.activation is 'sig':
            self.activations = np.vectorize(sigmoid)(activations)
            # print(self.activations)
        elif self.activation is 'none':
            self.activations = activations
            
            
    def update_dim(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.random.randn(self.output_dim, self.input_dim)/np.sqrt(self.input_dim + self.output_dim)


class InputLayer(Layer):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.activations = np.array([])
        
        
    def __str__(self):
        return f"InputLayer -> input_dim : {self.input_dim}"