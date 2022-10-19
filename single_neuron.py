# Reiznu Ahmad Tjandrida
# 21091397018

# Soal no 1a : Single Neuron

# Mengimport Library numpy, dan memberi inisial
import numpy as np

# Layer input 10 features
inputs = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

# Neuron 1
weights = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

bias = 4.0

outputs = np.dot(weights, inputs) + bias
print(outputs)
