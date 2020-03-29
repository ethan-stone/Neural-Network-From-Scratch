import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_der(x):
    # return sigmoid(x) * (1 - sigmoid(x))
    return x * (1 - x)


def relu(x):
    l = [0 if i < 0 else i for i in x]
    return np.array(l)
    
    
def relu_der(x):
    l = [0 if i < 0 else 1 for i in x]
    return np.array(l)