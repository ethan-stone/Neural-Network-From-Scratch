import numpy as np
from layer import Layer, InputLayer
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from activations import sigmoid

class NeuralNetwork():
    def __init__(self, eta=.01):
        self.layers = []
        self.eta = eta
        
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
        # print("feedforward")
        self.layers[0].activations = inputs
        for i in range(1, len(self.layers)):
            layer_i = self.layers[i]
            layer_i.feedforward(self.layers[i-1])
        
        output = self.layers[len(self.layers)-1].activations
        
        return output # return the output after the inputs feedforward through the netowrk
    
    
    def delta_error_delta_out(self, expected, predicted):
        return np.subtract(predicted, expected)
    
    
    def delta_out_delta_net(self, x):
        return sigmoid(x) * (1 - sigmoid(x))
    
    
    def delta_net_delta_weights(self, i):
        return self.layers[i-1].activations
    
    
    def backprop(self, expected):
        for i in range(len(self.layers)-1, 0, -1):
            layer_i = self.layers[i]
            predicted_i = layer_i.activations
            # print(predicted_i)
            delta_error_delta_out = self.delta_error_delta_out(expected, predicted_i)
            # print(delta_error_delta_out)
            delta_out_delta_net = self.delta_out_delta_net(predicted_i)
            # print(delta_out_delta_net)
            delta_net_delta_weights = self.delta_net_delta_weights(i)
            
            a = delta_error_delta_out * delta_out_delta_net
            # print(a)
            
            a_prime = a.reshape((-1, 1))
            
            b = delta_net_delta_weights
            
            c = a_prime * b
    
            layer_i.update_weights_biases(c, a, self.eta)
            
            
    def fit(self, data, labels, epochs=1):
        print("Training...")
        for epoch in range(epochs):
            i = 0
            while i < len(data):
                data_i = data[i]
                label_i = labels[i]
                self.feedforward(data_i)
                self.backprop(label_i)
                i+=1
            print("Epoch " + str(epoch))
        print("Finished training!")
        
    
    def predict(self, x):
        return self.feedforward(x)
    
    
    def evaluate(self, X, y):
        correct = 0
        print("Evaluating...")
        for i in range(0, len(X)):
            predicted = self.predict(X[i])
            error = self.error(y[i], predicted)
            if error < .05:
                correct += 1
        print("Finished evaluating!")
        return correct/len(X)
            
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

y = ohe.fit_transform(y.reshape((-1, 1)))

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

N = NeuralNetwork()
N.add_layer(InputLayer(784))
N.add_layer(Layer(10, activation='sig'))

N.fit(X_train, y_train, epochs=10)

for i in range(0, 100):
    predicted = N.predict(X_test[i])
    print(np.round(predicted))
    # print(y_test[i])
print("Finished evaluating!")

print(N.evaluate(X_test, y_test))


