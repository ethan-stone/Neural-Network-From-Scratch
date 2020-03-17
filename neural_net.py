class NeuralNetwork():
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        if len(self.layers) > 0:
            input_dim = layers[len(layers)-1].output_dim
            layer.update(input_dim)
            
        self.layers.append(layer)
    
        
    def feedforward(inputs):
        self.layers[0].activations = inputs
        for i in range(1, len(self.layers)):
            layer_i = layers[i]
            layer_i.feedforward(self.layers[i-1])
        
        output = self.layers[len(self.layers)].activations
        
        return output # return the output after the inputs feedforward throught the netowrk
    