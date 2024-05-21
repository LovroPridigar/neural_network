import numpy as np
import tensorflow as tf
from mnist import data

#Zloadamo testne primere
(images, label_train), (images_test, label_test) = data

#Število nevronov po slojih
list = [784, 32, 10]
size = len(list)

#Zloadamo shranjene uteži in biase
weights = []
bias = []
for i in range(size-1):
   weights.append(np.load(f'weights\Weights{i}.npy'))
   bias.append(np.load(f'weights\Bias{i}.npy'))

def sigmoid(z):
   return 1/(1 + np.exp(-z))

def dsigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Network(object):
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    #Za vektor dražljaja velikosti 729 vrne odgovor nevronske mreže
    def Eval(self, stimulus):
        a = []
        z = []
        for i in range(size-1):
            w, b = weights[i], bias[i]
            z.append(np.dot(w, stimulus) + b)
            stimulus = sigmoid(np.dot(w, stimulus) + b)
            a.append(stimulus)
        return a, z
    
    def Evaluate(self, stimulum):
        a, z = self.Eval(stimulum)
        return a[-1]
    
    def CostFunction(self, input, result):
        return np.sum(np.square(self.Evaluate(input) - result))
    
    def Gradient(self, stimulus, result):
        a, z = self.Eval(stimulus)
        w = [np.zeros((784, 32)), np.zeros((32, 10))]
        b = [np.zeros((32, 1)), np.zeros((10, 1))]
        for L in range(size-2, -1, -1):
            if L == 1:
                for j in range(0, 10):
                    b[L][j] = 2*(sigmoid(z[L][j]) - result[j]) * dsigmoid(z[j])
                    for k in range(0, 32):
                        w[L][k, j] = 2*(sigmoid(z[L][j]) - result[j]) * dsigmoid(z[j]) * a[L-1][k]
            else:
                pass
        return w, b

N = Network(weights, bias)
stim = images[0]
print(np.shape(label_train[0]))
print(N.Gradient(stim, label_train[0]))
