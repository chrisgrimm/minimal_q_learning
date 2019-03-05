import tensorflow as tf
import numpy as np
from reward_network import RewardPartitionNetwork
from gym import Env
from baselines.deepq.experiments.training_wrapper import make_dqn

class LinearAgent:

    def __init__(self, env : Env, rpn : RewardPartitionNetwork, name : str, gpu_num : int, reuse=None,
                 use_base=True):
        self.use_base = use_base
        if self.use_base:
            self.base_dqn = make_dqn(env, 'base_dqn', gpu_num)
        self.rpn = rpn
        self.N = rpn.num_partitions + (1 if self.use_base else 0)
        self.inp_Q_values_s = tf.placeholder(tf.float32, [None, env.action_space.n, self.N])
        self.inp_Q_values_sp = tf.placeholder(tf.float32, [None, env.action_space.n, self.N])
        self.inp_R = tf.placeholder(tf.float32, [None])
        self.inp_A = tf.placeholder(tf.int32, [None])
        self.lmbda = 0.1
        self.gamma = 0.99
        with tf.variable_scope(name, reuse=reuse) as vs:
            self.alpha = tf.get_variable(name, shape=[self.N+1],name='coeffs')
            self.Q_tilde_s = tf.reduce_sum(tf.reshape(self.alpha, [1,1,-1]) * self.inp_Q_values_s, axis=2)
            self.Q_tilde_sp = tf.reduce_sum(tf.reshape(self.alpha, [1,1,-1]) * self.inp_Q_values_sp, axis=2)
            Q = tf.reduce_sum(tf.one_hot(self.inp_A, env.action_space.n) * self.Q_tilde_s, axis=1)
            self.loss = tf.reduce_mean(tf.square(Q - (self.inp_R + self.gamma * tf.reduce_max(self.Q_tilde_sp, axis=1))), axis=0)
            self.reg = self.lmbda * tf.reduce_mean(tf.square(self.alpha))
            self.loss = self.loss + self.reg

            vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.original_name_scope)
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss, vars)
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.variables_initializer(vars))

    def get_action(self, s):
        Q_s = self.get_Q_values(s)
        return self.sess.run([self.Q_tilde_s], feed_dict={self.inp_Q_values_s: Q_s})[0]


    def get_Q_values(self, s):
        Qs = [qnet.get_Q(s) for qnet in self.rpn.Q_networks] + ([self.base_dqn.get_Q(s)] if self.use_base else [])
        return np.transpose(Qs, [1,2,0])


    def train(self, time, s, a, r, sp, t):
        if self.use_base:
            base_loss = self.train_base(time, s, a, r, sp, t)
        else:
            base_loss = -1
        Q_s = self.get_Q_values(s)
        Q_sp = self.get_Q_values(sp)
        [_, coeff_loss] = self.sess.run([self.train_op, self.loss], feed_dict={
            self.inp_Q_values_s: Q_s,
            self.inp_Q_values_sp: Q_sp,
            self.inp_R: r,
            self.inp_A: a
        })
        return coeff_loss, base_loss

    def train_base(self, time, s, a, r, sp, t):
        loss = self.base_dqn.train_batch(time, s, a, r, sp, t, np.ones_like(r), None)
        return loss
