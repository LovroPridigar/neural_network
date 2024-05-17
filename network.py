import numpy as np
list = [27*27, 32, 10] # število nevronov po slojih
size = len(list)

#zloadamo shranjene uteži in biase
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

    def evaluate(self, input_vector):
        for i in range(size-1):
            w, b = weights[i], bias[i]
            input_vector = sigmoid(np.dot(w, input_vector) + b)
        return input_vector
      

r = np.random.rand(27*27, 1)
print(Network().evaluate(r))