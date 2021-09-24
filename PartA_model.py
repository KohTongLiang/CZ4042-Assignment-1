import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
import pylab as plt
import multiprocessing as mp

# parameters
num_epochs = 50
seed = 100
rate = 0.001
batch_size = 1
dropout = 0.3
np.random.seed(seed)
tf.random.set_seed(seed)

class PartA_DNN():
    # init
    def __init__(self):
        # self.V = init_weights(num_hidden, num_labels)
        # self.c = init_bias(num_labels)
        # self.W = init_weights(num_features, num_hidden)
        # self.b = init_bias(num_hidden)
        self.num_epochs = 50
        self.seed = 100
        self.learning_rate = 0.001
        self.batch_size = 1
        self.dropout = 0.3
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # forward pass
    def __call__(self, x):
        z = tf.matmul(x, self.W) + self.b
        h = tf.nn.sigmoid(z)
        y = tf.matmul(h, self.V) + self.c
        return z, h, y
    
    def train(self, X, Y):
        model = Sequential([
            Dense(16, activation='relu'),
            Dense(1, activation='softmax')])
    
        model.compile(optimizer= Adam(learning_rate = self.learning_rate), loss='mse')
        
        history = model.fit(X, Y,
                            epochs = self.num_epochs, 
                            verbose = 2, 
                            use_multiprocessing = False)

        return(history)

    # loss function
    def loss(targets, outputs):
        return tf.reduce_mean(tf.reduce_sum(tf.square(targets - outputs),axis=1))