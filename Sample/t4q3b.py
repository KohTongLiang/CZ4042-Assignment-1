#
# Tutorial 4, Question 3b
#

import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

import numpy as np
import pylab as plt
import multiprocessing as mp

import os
if not os.path.isdir('figures'):
    os.makedirs('figures')
    
# parameters
num_epochs = 3000

seed = 100
np.random.seed(seed)
tf.random.set_seed(seed)


# generate data
X = np.zeros((9*9, 2)).astype(np.float32)
p = 0
for i in np.arange(-1, 1.001, 0.25):
    for j in np.arange(-1, 1.001, 0.25):
        X[p] = [i, j]
        p += 1
        
np.random.shuffle(X)
Y = np.zeros((9*9, 1)).astype(np.float32)
Y[:,0] = 0.8*X[:,0]**2 - X[:,1]**3 + 2.5*X[:,0]*X[:,1]


def my_train(rate):
    
    model = Sequential([
        Dense(10, activation='relu'),
        Dense(5, activation='relu'),
        Dense(1, activation='linear')])
    
   
    model.compile(optimizer= SGD(learning_rate = rate),
                  loss='mse')
    
    history = model.fit(X, Y, 
                        epochs=num_epochs, 
                        verbose = 2, 
                        use_multiprocessing=False)
    
                    
    return(history.history['loss'])


def main():

    rates = [0.001, 0.005, 0.01, 0.05]
    
    no_threads = mp.cpu_count()
    p = mp.Pool(processes = no_threads)
    costs = p.map(my_train, rates)

    plt.figure(1)
    for r in range(len(rates)):
      plt.plot(range(num_epochs), costs[r], label='lr = {}'.format(rates[r]))

    plt.xlabel('iterations')
    plt.ylabel('mean square error')
    plt.title('gradient descent learning')
    plt.legend()
    plt.savefig('./figures/t4q3b_1.png')

#    plt.show()


if __name__ == '__main__':
  main()

