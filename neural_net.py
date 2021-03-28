import numpy as np
from layer import Layer, InputLayer
from sklearn.datasets import fetch_openml, load_iris
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from activations import *
from tqdm import tqdm

class NeuralNetwork():
    def __init__(self, optimizer="adam", eta=.01, alpha=.9, beta1=.9, beta2=.999):
        self.layers = []
        self.optimizer = optimizer
        self.eta = eta
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        
        
    def add_layer(self, layer):
        if len(self.layers) > 0:
            input_dim = self.layers[len(self.layers)-1].output_dim
            layer.update_dim(input_dim)
            
        self.layers.append(layer)
        
        
    def error(self, expected, predicted):
        error = np.subtract(expected, predicted)
        squared = .5 * np.square(error)
        error = np.sum(squared)
        return error
    

    def feedforward(self, inputs):
        self.layers[0].activations = inputs
        for i in range(1, len(self.layers)):
            layer_i = self.layers[i]
            layer_i.feedforward(self.layers[i-1])
        
        output = self.layers[len(self.layers)-1].activations
        
        return output # return the output after the inputs feedforward through the network
    
    
    def calc_layer_errors(self, e_out):
        for i in range(len(self.layers)-1, 0, -1):
            layer_i = self.layers[i]
            if isinstance(layer_i, ConvalutionLayer):
                prev_layer = self.layers[i+1]
                if isinstance(prev_layer, ConvalutionLayer):
                    prev_kernel_size = prev_layer.kernel_size
                    prev_kernel = prev_layer.kernel
                    prev_cache = prev_layer.cache
                    prev_error = prev_layer.error
                    prev_activations = layer_i.activations
                    for m in range(layer_i.output_dim):
                        for n in range(layer_i.output_dim):
                            for a in range(prev_kernel_size):
                                for b in range (prev_kernel_size):
                                    layer_i.error[m, n] = prev_error[m-a, n-b]*sigmoid_der(prev_activations[m-a, n-b])*prev_kernel[a, b]
                      
                elif isinstance(prev_layer, Layer):
                    prev_error = prev_layer.error
                    prev_weights = prev_layer.weights
                    sum_weights = prev_weights.sum(axis=1)
                    a = prev_weights.transpose()
                    b = np.dot(a, prev_errors)
                    layer_i.error = b.reshape(layer_i.output_dim, layer_i.output_dim)       
                
            elif isinstance(layer_i, Layer):
                if i == len(self.layers)-1:
                    layer_i.error = e_out
                else:
                    prev_layer = self.layers[i+1]
                    prev_weights = prev_layer.weights
                    prev_errors = prev_layer.error
                    sum_weights = prev_weights.sum(axis=1)
                    a = prev_weights.transpose()
                    
                    b = np.dot(a, prev_errors)
                    
                    layer_i.error = b
    
    
    def backprop(self):
        for i in range(len(self.layers)-1, 0, -1):
            layer_i = self.layers[i]
            predicted_i = layer_i.activations
            cache_i = layer_i.cache
            error_i = layer_i.error
            activation_i = layer_i.activation
            
            if activation_i is "sig":
                delta_weights = np.dot((-error_i * sigmoid_der(predicted_i)).reshape((-1, 1)), self.layers[i-1].activations.reshape((1, -1)))
                delta_biases = -error_i * sigmoid_der(predicted_i)
            elif activation_i is "relu":
                delta_weights = np.dot((-error_i * relu_der(cache_i)).reshape((-1, 1)), self.layers[i-1].activations.reshape((1, -1)))
                delta_biases = -error_i * relu_der(cache_i)
            elif activation_i is "leaky":
                delta_weights = np.dot((-error_i * leaky_relu_der(cache_i)).reshape((-1, 1)), self.layers[i-1].activations.reshape((1, -1)))
                delta_biases = -error_i * leaky_relu_der(cache_i)
            elif activation_i is "linear":
                delta_weights = np.dot((-error_i * linear(cache_i)).reshape((-1, 1)), self.layers[i-1].activations.reshape((1, -1)))
                delta_biases = -error_i * linear_der(cache_i)
                               
            layer_i.add_gradient(delta_weights)
            
            if self.optimizer is "adam":
                layer_i.adam(delta_weights, delta_biases, self.beta1, self.beta2, self.eta)
            elif self.optimizer is "rmsprop":
                layer_i.rmsprop(delta_weights, delta_biases, self.eta, self.alpha)
            elif self.optimizer is "grad":
                layer_i.grad(delta_weights, delta_biases, self.eta)
            
            
    def fit(self, data, labels, epochs=1):
        print("Training...")
        for epoch in range(epochs):
            print("Epoch " + str(epoch + 1))
            for i in tqdm(range(len(data))):
                data_i = data[i]
                label_i = labels[i]
                out = self.feedforward(data_i)
                self.calc_layer_errors(label_i - out)
                self.backprop()
        print("Finished training!")
        
    
    def predict(self, x):
        return self.feedforward(x)
    
    
    def evaluate(self, X, y):
        correct = 0
        print("Evaluating...")
        for i in range(0, len(X)):
            predicted = self.predict(X[i])
            max_value = np.amax(predicted)
            predicted = np.array([0 if a < max_value else 1 for a in predicted])
            
            if np.array_equal(predicted, y[i]):
                correct += 1
        print("Finished evaluating!")
        return correct/len(X)
    
            
            
# ohe = OneHotEncoder(sparse=False)

# mnist = fetch_openml('mnist_784', version=1)
# X, y = mnist["data"], mnist["target"]

# y = ohe.fit_transform(y.reshape((-1, 1)))

# X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)

# iris = load_iris()

# X = iris.data
# y = iris.target

# y = ohe.fit_transform(y.reshape((-1, 1)))

# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=.5)

# def f(x):
#     return x**2 + 2*x + 3

# X = []
# y = []
# for i in range(-1000, 1000, 2):
#     a = np.random.rand()/10
#     X.append(np.array([i*a]).reshape(1, ))
#     y.append(np.array([f(i*a)]).reshape(1, ))
    
# X = np.array(X)
# y = np.array(y)

# N = NeuralNetwork(eta=.001, optimizer="adam")
# N.add_layer(InputLayer(784))
# N.add_layer(Layer(128, activation='leaky'))
# N.add_layer(Layer(32, activation='leaky'))
# N.add_layer(Layer(16, activation='leaky'))
# N.add_layer(Layer(10, activation='sig'))

# N.fit(X_train, y_train, epochs=10)

# # for i in range(-100, 100, 3):
# #     print(str(N.predict(np.array([i]).reshape(1, ))) + " " + str(f(i)))

# print(N.evaluate(X_test, y_test))

network = NeuralNetwork()
network.add_layer(InputLayer(3))
network.add_layer(Layer(2))
network.add_layer(Layer(1))

result = network.feedforward(np.array([1, 1, 1]))
print(result)



