import numpy as np
from activations import *

input_dim = 5
kernel_dim = 3
step = 1
output_dim = int((input_dim - kernel_dim)/1 + 1)

A = np.random.randn(input_dim, input_dim)
kernel = np.random.randn(kernel_dim, kernel_dim)
output = np.zeros((output_dim, output_dim))

for i in range(output_dim):
    for j in range(output_dim):
        chunk = A[i:i+kernel_dim,j:j+kernel_dim]
        feature = sigmoid(np.multiply(kernel, chunk).sum())
        output[i, j] = feature
    