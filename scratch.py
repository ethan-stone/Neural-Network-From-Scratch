import numpy as np

A = np.array([1, 2, 3, 4, 5, 6])
b = np.array([1, 2, 3, 4, 5, 6])

max_value = np.amax(A)

a = np.array([0 if a < max_value else 1 for a in A])

print(a)