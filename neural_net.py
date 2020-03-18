import numpy as np
from layer import Layer

class NeuralNetwork():
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        if len(self.layers) == 0:
            input_dim = layer.output_dim
            layer.update_dim(input_dim)

        if len(self.layers) > 0:
            input_dim = self.layers[len(self.layers)-1].output_dim
            layer.update_dim(input_dim)
            
        self.layers.append(layer)
        
        
    def total_error(self, expected, predicted):
        error = np.subtract(expected, predicted)
        squared = .5*np.square(error)
        total_error = np.sum(squared)
        
        
    def feedforward(self, inputs):
        self.layers[0].activations = inputs
        for i in range(1, len(self.layers)):
            layer_i = self.layers[i]
            layer_i.feedforward(self.layers[i-1])
        
        output = self.layers[len(self.layers)-1].activations
        
        return output # return the output after the inputs feedforward throught the netowrk
    

# N = NeuralNetwork()
# N.add_layer(Layer(16))
# N.add_layer(Layer(4))
# N.add_layer(Layer(2))

# output = N.feedforward(np.random.rand(16))

# print(output)