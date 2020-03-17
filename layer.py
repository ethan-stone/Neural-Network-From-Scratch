from neuron import Neuron

class Layer():
    def __init__(self, input_dim, output_dim):
        """
        Constructor
        
        input_dim : number of inputs for the layer (optional)
        output_dim : number of outputs for the layer, essentially the number of neurons in the layer
        
        """
        self.neurons = [Neuron() for i in range(0, output_dim)]
