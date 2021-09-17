import tensorflow as tf

class PartA_DNN():
    # init
    def __init__(self, num_features, num_hidden, num_labels):
        self.V = init_weights(num_hidden, num_labels)
        self.c = init_bias(num_labels)
        self.W = init_weights(num_features, num_hidden)
        self.b = init_bias(num_hidden)

    # forward pass
    def __call__(self, x):
        z = tf.matmul(x, self.W) + self.b
        h = tf.nn.sigmoid(z)
        y = tf.matmul(h, self.V) + self.c
        return z, h, y
    
    # loss function
    def loss(targets, outputs):
        return tf.reduce_mean(tf.reduce_sum(tf.square(targets - outputs),axis=1))