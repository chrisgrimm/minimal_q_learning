import numpy as np
import tensorflow as tf
import os


class ReparameterizedRewardNetwork(object):

    def __init__(self, num_rewards, learning_rate, buffer, num_actions, name, reuse=None):
        self.buffer = buffer
        self.num_rewards = self.num_partitions = num_rewards
        self.num_actions = num_actions
        self.gamma = 0.99
        self.inp_s = tf.placeholder(tf.uint8, [None, 64, 64, 3])
        self.converted_inp_s = tf.image.convert_image_dtype(self.inp_s, tf.float32)
        self.inp_a = tf.placeholder(tf.int32, [None])
        self.inp_r = tf.placeholder(tf.float32, [None])
        self.inp_sp = tf.placeholder(tf.uint8, [None, 64, 64, 3])
        self.converted_inp_sp = tf.image.convert_image_dtype(self.inp_sp, tf.float32)

        with tf.variable_scope(name, reuse=reuse) as scope:
            self.Q_s, self.Q_sp, self.R = self.setup_Q_functions()
            (self.sums_to_R, self.greater_than_0, self.reward_consistency,
                self.J_indep, self.J_nontriv) = self.setup_constraints(self.Q_s, self.Q_sp, self.R)

            self.loss = 10000*self.sums_to_R + 10000*self.greater_than_0 + 10000*self.reward_consistency + self.J_indep - 0.1*self.J_nontriv
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            self.variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.original_name_scope)


        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        self.saver = tf.train.Saver(var_list=self.variables)
        self.sess.run(tf.variables_initializer(self.variables))

    def save(self, path, name):
        self.saver.save(self.sess, os.path.join(path, name))


    def restore(self, path, name):
        self.saver.restore(self.sess, os.path.join(path, name))


    def train_R_functions(self):
        batch_size = 32
        S, A, R, SP, T = self.buffer.sample(batch_size)
        [_, sums_to_R, greater_than_0, reward_consistency, J_indep, J_nontriv] = self.sess.run(
            [self.train_op, self.sums_to_R, self.greater_than_0, self.reward_consistency, self.J_indep, self.J_nontriv],
                      feed_dict={self.inp_s: S, self.inp_a: A, self.inp_r: R, self.inp_sp: SP})
        return sums_to_R, greater_than_0, reward_consistency, J_indep, J_nontriv

    def get_partitioned_reward(self, s, a, sp):
        return np.transpose(self.sess.run([self.R[(i,i)] for i in range(self.num_rewards)],
                            feed_dict={self.inp_s: s, self.inp_a: a, self.inp_sp: sp}), [1,0]) # [bs, num_rewards]


    def get_state_values(self, s):
        Qs = self.sess.run([self.Q_s[(i,i)] for i in range(self.num_rewards)], feed_dict={self.inp_s: s})
        return [np.max(Q, axis=1) for Q in Qs] # [num_partitions, bs]


    def get_state_actions(self, s):
        Qs = self.sess.run([self.Q_s[(i,i)] for i in range(self.num_rewards)], feed_dict={self.inp_s: s})
        return [np.argmax(Q, axis=1) for Q in Qs]

    def get_hybrid_actions(self, s, mode='sum'):
        pre_hybrid = self.sess.run([self.Q_s[(i,i)] for i in range(self.num_rewards)], feed_dict={self.inp_s: s})
        if mode == 'sum':
            hybrid_q = np.sum(pre_hybrid, axis=0)  # [bs, num_actions]
        elif mode == 'max':
            hybrid_q = np.max(pre_hybrid, axis=0)
        else:
            raise Exception(f'Unrecognized mode: {mode}')
        return np.argmax(hybrid_q, axis=1)  # [bs]

    def get_reward(self, s, a, sp):

        return self.get_partitioned_reward([s], [a], [sp])[0]






    def build_Q_network(self, s, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            c0 = tf.layers.conv2d(s, 32, 8, 4, 'SAME', activation=tf.nn.relu, name='c0')  # [bs, 16, 16, 32]
            c1 = tf.layers.conv2d(c0, 64, 4, 2, 'SAME', activation=tf.nn.relu, name='c1')  # [bs, 8, 8, 64]
            c2 = tf.layers.conv2d(c1, 64, 3, 1, 'SAME', activation=tf.nn.relu, name='c2')  # [bs, 8, 8, 64]
            internal_rep = tf.reshape(c2, [-1, 8 * 8 * 64])
            fc1 = tf.layers.dense(internal_rep, 128, activation=tf.nn.relu, name='fc1')
            q = tf.layers.dense(fc1, self.num_actions, name='q')
            return q


    def setup_Q_functions(self):
        # build necessary q terms
        Q_s, Q_sp = dict(), dict()
        for i in range(self.num_rewards):
            for j in range(self.num_rewards):
                Q_ij_s = self.build_Q_network(self.converted_inp_s, f'Q_{i}_{j}')
                Q_s[(i,j)] = Q_ij_s
                Q_ij_sp = self.build_Q_network(self.converted_inp_sp, f'Q_{i}_{j}', reuse=True)
                Q_sp[(i,j)] = Q_ij_sp
        # build reward terms
        R = dict()
        for i in range(self.num_rewards):
            for j in range(self.num_rewards):
                first_term = tf.reduce_sum(tf.one_hot(self.inp_a, self.num_actions) * Q_s[(i,j)], axis=1)
                if i == j:
                    print(Q_sp[(i,j)])
                    second_term = self.gamma * tf.reduce_max(Q_sp[(i,j)], axis=1)
                else:
                    action_choice = tf.stop_gradient(tf.argmax(Q_sp[(j,j)],axis=1))
                    action_choice = tf.one_hot(action_choice, self.num_actions)
                    second_term = self.gamma * tf.reduce_sum(Q_sp[(i,j)] * action_choice, axis=1)
                R_ij = first_term - second_term
                R[(i,j)] = R_ij
        return Q_s, Q_sp, R

    def setup_constraints(self, Q_s, Q_sp, R):
        # set up reward_constraints
        sums_to_R = tf.reduce_mean(tf.square(tf.reduce_sum([R[(i,i)] for i in range(self.num_rewards)], axis=0) - self.inp_r), axis=0)
        greater_than_0 = tf.reduce_mean(tf.reduce_sum([tf.square(tf.maximum(0.0, -R[(i,i)])) for i in range(self.num_rewards)], axis=0), axis=0)

        # set up consistency constraints
        reward_consistency = 0
        for i in range(self.num_rewards):
            for j in range(self.num_rewards):
                reward_consistency += tf.reduce_mean(tf.square(R[(i,i)] - R[(i,j)]), axis=0)

        # set up J_indep, J_nontriv
        J_indep = 0
        J_nontriv = 0
        for i in range(self.num_rewards):
            for j in range(self.num_rewards):
                if i == j:
                    V_ii = tf.reduce_mean(tf.reduce_max(Q_s[(i,j)], axis=1), axis=0)
                    J_nontriv += V_ii
                else:
                    pi_j_action = tf.one_hot(tf.argmax(Q_s[(i,j)], axis=1), self.num_actions)
                    V_ij = tf.reduce_mean(tf.reduce_sum(Q_s[(i,j)] * pi_j_action, axis=1), axis=0)
                    J_indep += V_ij
        return sums_to_R, greater_than_0, reward_consistency, J_indep, J_nontriv


