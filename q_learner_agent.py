import tensorflow as tf
import numpy as np


class QLearnerAgent(object):

    def __init__(self, obs_size, num_actions):
        print(obs_size, num_actions)
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.gamma = 0.99

        self.inp_s = tf.placeholder(tf.float32, [None, obs_size], name='inp_s')
        self.inp_a = tf.placeholder(tf.int32, [None], name='inp_a')
        inp_a_onehot = tf.one_hot(self.inp_a, self.num_actions)
        self.inp_r = tf.placeholder(tf.float32, [None], name='inp_r')
        self.inp_t = tf.placeholder(tf.float32, [None], name='inp_t')
        self.inp_sp = tf.placeholder(tf.float32, [None, obs_size], name='inp_sp')

        self.qa = qa = self.qa_network(self.inp_s, 'qa_network')
        qa_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='qa_network/')

        qa_target = self.qa_network(self.inp_sp, 'qa_target')
        qa_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='qa_target/')

        y = self.inp_r + (1 - self.inp_t) * self.gamma * tf.reduce_max(qa_target, axis=1)
        self.loss = tf.reduce_mean(tf.square(tf.stop_gradient(y) - tf.reduce_sum(qa * inp_a_onehot, axis=1)))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

        # variable update operations
        tau = 0.99
        soft_update_target = tf.group(*[tf.assign(target, tau*target + (1 - tau)*network)
                                        for network, target in zip(qa_vars, qa_target_vars)])
        self.soft_update_target = soft_update_target
        hard_update_target = tf.group(*[tf.assign(target, network)
                                        for network, target in zip(qa_vars, qa_target_vars)])
        self.hard_update_target = hard_update_target

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.hard_update_target)


    def qa_network(self, obs, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            fc1 = tf.layers.dense(obs, 128, tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 128, tf.nn.relu, name='fc2')
            qa = tf.layers.dense(fc2, self.num_actions, name='qa')
            return qa

    def train_batch(self, s, a, r, sp, t):
        [_, loss] = self.sess.run([self.train_op, self.loss], feed_dict={
            self.inp_s: s,
            self.inp_a: a,
            self.inp_r: r,
            self.inp_sp: sp,
            self.inp_t: t
        })
        self.sess.run([self.soft_update_target])
        return loss

    def get_action(self, s):
        [qa] = self.sess.run([self.qa], feed_dict={self.inp_s: s})
        a = np.argmax(qa, axis=1)
        return a



