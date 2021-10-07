import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam

import numpy as np
import pylab as plt
import multiprocessing as mp

# parameters
# num_epochs = 50
# seed = 100
# rate = 0.001
# batch_size = 1
# dropout = 0.3
# np.random.seed(seed)
# tf.random.set_seed(seed)

def init_bias(n = 1):
        return(tf.Variable(np.zeros(n), dtype=tf.float64))

def init_weights(n_in=2, n_out=1, logistic=True):
    W_values = np.asarray(np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                            high=np.sqrt(6. / (n_in + n_out)),
                                            size=(n_in, n_out)))
    if logistic == True:
        W_values *= 4
    return(tf.Variable(W_values, dtype=tf.float64))
    
def loss(targets, outputs):
    return tf.reduce_mean(tf.reduce_sum(tf.square(targets - outputs),axis=1))

class PartA_DNN():
    # init
    def __init__(self,
        learning_rate = 0.001,
        batch_size = 1,
        dropout = 0.3,
        seed = 100,
        num_epochs = 50
        ):

        # self.W = init_weights(n_in = 1, n_out = 16)
        # self.b = init_bias(16)
        # self.V = init_weights(n_in = 16, n_out = 1)
        # self.c = init_bias(1)
        self.seed = seed
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_epochs = num_epochs
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.model = Sequential([
            Dense(16, activation='relu'),
            Dropout(rate = self.dropout),
            Dense(1, activation='softmax')
        ])

    # forward pass
    # def __call__(self, x, y):
    #     z = tf.matmul(x, self.W) + self.b
    #     h = tf.nn.relu(z)
    #     y = tf.matmul(h, self.V) + self.c
    #     out = tf.nn.softmax(y)
    #     return z, h, y, out
    
    def train(self, X_train, Y_train, X_test, Y_test):
        self.model.compile(optimizer= Adam(learning_rate = self.learning_rate), loss='mse')
        
        history = self.model.fit(X_train, Y_train,
                            epochs = self.num_epochs, 
                            verbose = 2, 
                            use_multiprocessing = False,
                            validation_data=(X_test, Y_test)
                            )
        return history
    
    def view_summary(self):
        return self.model.summary()
