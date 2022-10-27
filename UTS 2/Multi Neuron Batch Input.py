#Diego Athalla Samudero
#21091397042

#mengimport numpy library sebagai np
import numpy as np

#menginisialisasi input
inputs = [  [1.4, 3.4, 3.4, 1.6, 2.8, 3.9, 4.4, 5.9, 6.7, 7.9],
            [2.5, 1.7, 8.3, 6.5, 7.6, 5.6, 8.5, 1.9, 3.9, 2.7],
            [1.3, 8.5, 7.6, 9.8, 4.5, 5.8, 4.7, 0.9, 8.8, 7.7],
            [2.1, 3.2, 4.1, 1.1, 6.2, 7.2, 5.9, 9.1, 4.3, 6.5],
            [3.3, 4.4, 6.3, 1.7, 1.3, 2.6, 7.3, 8.2, 6.2, 3.1],
            [5.8, 2.5, 1.4, 5.8, 3.1, 0.2, 6.3, 6.9, 7.9, 5.8]]

#panjang weights1
weights1 = [[1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 1.5, 2.6, 3.1, 4.2],
            [7.8, 8.9, 9.1, 2.4, 2.5, 7.5, 7.3, 6.7, 5.5, 2.5],
            [9.3, 1.2, 5.2, 4.9, 9.8, 3.7, 1.4, 2.1, 5.6, 7.7],
            [2.6, 7.5, 5.4, 4.3, 3.2, 2.1, 5.7, 6.1, 2.2, 1.1],
            [4.4, 5.4, 5.9, 6.1, 9.2, 8.3, 3.6, 6.3, 2.1, 0.3]]

#jumlah biases pada layer1
biases1 = [1.4, 9.3, 2.5, 4.9, 2.9]

#panjang weight2
weights2 =  [   [0.1, 3.2, 4.4, 6.4, 8.6],
                [4.2, 1.1, 1.5, 7.8, 1.5],
                [9.6, 8.4, 6.8, 3.7, 2.4]]

#jumlah biases pada layer2
biases2= [3.4, 2.1, 1.7]

#menghitung layer1 menggunakan inputs, weights1, dan biases1
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

#menghitung layer2 dari hasil perhitungan layer1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

#print output layer2
print(layer2_outputs)