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
        self.weights = np.random.randn(self.output_dim, self.input_dim)
        self.biases = np.zeros(output_dim)
        self.cache = 0
        self.error = np.zeros(output_dim)
        self.gradients = np.zeros((self.output_dim, self.input_dim))
        self.epsilon = np.full((self.output_dim, self.input_dim), 1)*(10**(-8))
        self.prev_E = 0
        self.prev_m = 0
        self.prev_v = 0
        self.t = 1
        
        
    def __str__(self):
        return f"Layer -> input_dim : {self.input_dim}, output_dim : {self.output_dim}"
    
    
    def update_dim(self, input_dim):
        self.input_dim = input_dim
        if self.activation is "sig":
            self.weights = np.random.randn(self.output_dim, self.input_dim)*(4*np.sqrt(6/(self.input_dim + self.output_dim)))
        elif self.activation is "relu":
            self.weights = np.random.randn(self.output_dim, self.input_dim)*(np.sqrt(2)*np.sqrt(6/(self.input_dim + self.output_dim)))
        self.gradients = np.zeros((self.output_dim, self.input_dim))
        self.epsilon = np.full((self.output_dim, self.input_dim), 1)*(10**(-8))
    

    def add_gradient(self, gradient):
        self.gradients = self.gradients + gradient**2
        
    
    def E(self, x, alpha):
        E = (1-alpha)*x**2 + alpha*self.prev_E
        self.prev_E = E
        return E
        
        
    def rmsprop(self, delta_weights, delta_biases, eta, alpha):
        self.weights = self.weights - (eta * delta_weights)/np.sqrt(self.epsilon + self.E(delta_weights, alpha))
        self.biases = self.biases - (eta * delta_biases)
        
        
    def adam(self, delta_weights, delta_biases, beta1, beta2, eta):
        m = beta1 * self.prev_m + (1 - beta1) * delta_weights
        v = beta2 * self.prev_v + (1 - beta2) * (delta_weights**2)
        
        self.prev_m = m
        self.prev_v = v
        
        m_hat = m/(1 - beta1**self.t)
        v_hat = v/(1 - beta2**self.t)
        
        self.t += 1
        self.weights = self.weights - (eta * m_hat)/(self.epsilon + np.sqrt(v_hat))
        self.biases = self.biases - (eta * delta_biases)
    
    
    def grad(self, delta_weights, delta_biases, eta):
        self.weights = self.weights - (eta * delta_weights)
        self.biases = self.biases - (eta * delta_biases)
        
        
    def feedforward(self, input_layer):
        input_activations = input_layer.activations
        
        dot_product = np.dot(self.weights, input_activations)
        activations = np.add(dot_product, self.biases)
        self.cache = activations
        
        if self.activation is 'sig':
            self.activations = sigmoid(activations)
        elif self.activation is 'relu':
            self.activations = relu(activations)



class InputLayer(Layer):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.activations = np.array([])
        
        
    def __str__(self):
        return f"InputLayer -> input_dim : {self.input_dim}"