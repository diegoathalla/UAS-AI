# Reiznu Ahmad Tjandrida
# 21091397018

# Soal no 1b : Multi Neuron

# Mengimport Library numpy, dan memberi inisial
import numpy as np

# Layer input 10 features
inputs = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

# Multi Neuron
weights = [
    [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    [2.7, 1.8, 2.6, 2.8, 3.6, 3.8, 4.6, 4.8, 5.6, 5.8],
    [1.0, 1.5, 2.0, 2.5, 3.7, 3.5, 4.7, 4.5, 5.0, 5.5],
    [1.5, 1.4, 2.2, 2.4, 3.2, 3.4, 4.2, 4.4, 5.2, 5.4],
    [2.7, 1.8, 2.6, 2.8, 3.6, 3.8, 4.6, 4.8, 5.6, 5.8]
]

biases = [1.5, 2.3, 5.6, 9.8, 10.9]

outputs = np.dot(weights, inputs) + biases
print(outputs)
