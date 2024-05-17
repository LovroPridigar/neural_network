import numpy as np
list = [784, 32, 10] # število nevronov po slojih
size = len(list)

#inicializiramo uteži in biase
weights = [np.random.rand(y, x) for x, y in zip(list[:-1], list[1:])]
bias = [np.random.rand(x, 1) for x in list[1:]] 

for i in range(size-1):
   np.save(f'weights\Weights{i}.npy', weights[i])
   np.save(f'weights\Bias{i}.npy', bias[i])