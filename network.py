import numpy as np
import tensorflow as tf

#Pripravimo primere za treniraje in testiranje
mnist = tf.keras.datasets.mnist
#(primeri za trening), (primeri za testiranje)
(stim_train, sol_train), (stim_test, sol_test) = mnist.load_data()

#Normaliziramo vhodne vektorje, na začetku so elementi števila med 0-250
stim_train = tf.keras.utils.normalize(stim_train, axis=1)
stim_test = tf.keras.utils.normalize(stim_test, axis=1)

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

class Network(object):
    def __init__(self):
        self.weights = weights
        self.bias = bias
    
    #Za vektor dražljaja velikosti 729 vrne odgovor nevronske mreže
    def Evaluate(self, stimulus):
        for i in range(size-1):
            w, b = weights[i], bias[i]
            stimulus = sigmoid(np.dot(w, stimulus) + b)
        return stimulus
    
    def CostFunction(self, input, result):
        return np.sum(np.square(self.Evaluate(input) - result))
    
