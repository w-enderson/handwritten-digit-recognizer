import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.datasets import mnist
import numpy as np

# Carregar os dados
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("train ", X_train.shape) 
print("test ", X_test.shape) 

class Neuron:
    def __init__(self, x, w, b):
        self.inputs = x
        self.weights = w
        self.bias = b

    def ReLU(self, x):
        return  max(0, x)
    
    def getOutput(self):
        z = np.dot(self.inputs, self.weights) + self.bias
        return self.ReLU(z)

# inputs = np.array([1.0, 2.0, 3.0])
# weights = np.ones(3, dtype=int)
# bias = 0

# neuron = Neuron(inputs, weights, bias)
# print(neuron.getOutput()) 

class NeuralNetwork:
    def __init__(self, neuronsLayer):
        self.neuronsLayer = neuronsLayer

    def create(self):
        pass





