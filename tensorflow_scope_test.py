import tensorflow as tf
import numpy as np

def simple_func(x, name ,reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        y = tf.layers.dense(x, 10, activation=tf.nn.relu, name='fc1')
    return y

x = tf.placeholder(tf.float32, [None, 10])

y_A = simple_func(x, 'A')

with tf.variable_scope('B'):
    y_BA = simple_func(x, 'A')
    # this is how you do it
    y_A_again = simple_func(x, tf.VariableScope(True, name='A'))
vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(vars)

