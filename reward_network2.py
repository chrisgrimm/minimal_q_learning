import numpy as np
import tensorflow as tf
import os
from baselines.deepq.experiments.training_wrapper import make_dqn


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculates the huber loss.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    err = tf.abs(y_true - y_pred, name='abs')
    mg = tf.constant(max_grad, name='max_grad')
    lin = mg*(err-.5*mg)
    quad=.5*err*err
    return tf.where(err < mg, quad, lin)

class ReparameterizedRewardNetwork(object):

    def __init__(self, env, num_rewards, learning_rate, buffer, num_actions, name, num_channels=3, gpu_num=-1, reuse=None):
        self.buffer = buffer
        self.num_rewards = self.num_partitions = num_rewards
        self.num_actions = num_actions
        self.gamma = 0.99

        self.inp_s = tf.placeholder(tf.uint8, [None, 64, 64, num_channels])
        self.converted_inp_s = tf.image.convert_image_dtype(self.inp_s, tf.float32)
        self.inp_a = tf.placeholder(tf.int32, [None])
        self.inp_r = tf.placeholder(tf.float32, [None])
        self.inp_sp = tf.placeholder(tf.uint8, [None, 64, 64, num_channels])
        self.converted_inp_sp = tf.image.convert_image_dtype(self.inp_sp, tf.float32)
        self.use_target = True
        self.use_shared_q_repr = True
        self.use_huber = True
        self.seperate_policy_networks = True
        self.enforce_random_subset = True

        self.batch_size = 32

        self.dqn = make_dqn(env, f'qnet', gpu_num=gpu_num, multihead=True, num_heads=num_rewards)



        with tf.variable_scope(name, reuse=reuse) as scope:
            self.Q_s, self.Q_sp, self.R, self.soft_update, self.hard_update = self.setup_Q_functions()
            (self.sums_to_R, self.greater_than_0, self.reward_consistency,
                self.J_indep, self.J_nontriv) = self.setup_constraints(self.Q_s, self.Q_sp, self.R)

            self.loss = 10000*(self.sums_to_R + self.greater_than_0 + self.reward_consistency) + self.J_indep - 10*self.J_nontriv
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            self.variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.original_name_scope)


        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        self.saver = tf.train.Saver(var_list=self.variables)
        self.sess.run(tf.variables_initializer(self.variables))
        if self.use_target:
            self.sess.run(self.hard_update)


    def save(self, path, name):
        self.dqn.save(path, 'qnet.ckpt')
        self.saver.save(self.sess, os.path.join(path, name))


    def restore(self, path, name):
        self.dqn.restore(path, 'qnet.ckpt')
        self.saver.restore(self.sess, os.path.join(path, name))


    def train_R_functions(self, time):
        q_loss = self.train_Q_functions(time)
        S, A, R, SP, T = self.buffer.sample(self.batch_size)
        [_, sums_to_R, greater_than_0, reward_consistency, J_indep, J_nontriv] = self.sess.run(
            [self.train_op, self.sums_to_R, self.greater_than_0, self.reward_consistency, self.J_indep, self.J_nontriv],
                      feed_dict={self.inp_s: S, self.inp_a: A, self.inp_r: R, self.inp_sp: SP})
        if self.use_target:
            self.sess.run(self.soft_update)
        return sums_to_R, greater_than_0, reward_consistency, J_indep, J_nontriv, q_loss

    def train_Q_functions(self, time):
        S, A, R, SP, T = self.buffer.sample(self.batch_size)
        reward_num_samples = np.random.randint(0, self.num_rewards, size=[self.batch_size])
        rewards = self.get_partitioned_reward(S, A, SP) # [bs, num_rewards]
        selected_rewards = rewards[range(self.batch_size), reward_num_samples] # [bs]
        weights, batch_idxes = np.ones_like(T), None
        loss = self.dqn.train_batch(time, S, A, selected_rewards, SP, T, weights, batch_idxes, reward_num_samples)
        return loss




    def get_partitioned_reward(self, s, a, sp):
        return np.transpose(self.sess.run([self.R[(i,i)] for i in range(self.num_rewards)],
                            feed_dict={self.inp_s: s, self.inp_a: a, self.inp_sp: sp}), [1,0]) # [bs, num_rewards]


    def get_state_values(self, s):
        x = self.dqn.get_all_Q(s) # [bs, num_actions, num_rewards]
        x = np.max(x, axis=1) # [bs, num_rewards]
        x = np.transpose(x, [1,0]) #[num_rewards, bs]
        return x



    def get_state_actions(self, s):
        x = self.dqn.get_all_Q(s) # [bs, num_actions, num_rewards]
        x = np.argmax(x, axis=1) # [bs, num_rewards]
        x = np.transpose(x, [1,0]) # [num_rewards, bs]
        return x
        #Qs = self.sess.run([self.Q_s[(i,i)] for i in range(self.num_rewards)], feed_dict={self.inp_s: s})
        #return [np.argmax(Q, axis=1) for Q in Qs]

    def get_Qs(self, s):
        x = self.dqn.get_all_Q(s) # [bs, num_actions, num_rewards]
        x = np.transpose(x, [2, 0, 1]) # [num_rewards, bs, num_actions]
        return x
        #x = [self.dqn.get_Q(s, [reward_num]*len(s)) for reward_num in range(self.num_rewards)] # [num_rewards, bs, num_actions]
        #return x
        #Qs = self.sess.run([self.Q_s[(i,i)] for i in range(self.num_rewards)], feed_dict={self.inp_s: s})
        #return Qs


    def get_hybrid_actions(self, s, mode='sum'):
        x = self.get_Qs(s) # [num_rewards, bs, num_actions]
        pre_hybrid = x
        if mode == 'sum':
            hybrid_q = np.sum(pre_hybrid, axis=0)  # [bs, num_actions]
        elif mode == 'max':
            hybrid_q = np.max(pre_hybrid, axis=0)
        else:
            raise Exception(f'Unrecognized mode: {mode}')
        return np.argmax(hybrid_q, axis=1)  # [bs]


    def get_reward(self, s, a, sp):
        return self.get_partitioned_reward([s], [a], [sp])[0]


    def build_shared_Q_network_trunk(self, s, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            c0 = tf.layers.conv2d(s, 32, 8, 4, 'SAME', activation=tf.nn.relu, name='c0')  # [bs, 16, 16, 32]
            c1 = tf.layers.conv2d(c0, 64, 4, 2, 'SAME', activation=tf.nn.relu, name='c1')  # [bs, 8, 8, 64]
            c2 = tf.layers.conv2d(c1, 64, 3, 1, 'SAME', activation=tf.nn.relu, name='c2')  # [bs, 8, 8, 64]
            internal_rep = tf.reshape(c2, [-1, 8 * 8 * 64])
            fc1 = tf.layers.dense(internal_rep, 128, activation=tf.nn.relu, name='fc1')
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.original_name_scope)
        return fc1, vars

    def build_shared_Q_network_head(self, fc1, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            q = (1 / (1 - self.gamma)) * tf.layers.dense(fc1, self.num_actions, activation=tf.nn.sigmoid, name='q')
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.original_name_scope)
        return q, vars

    def build_Q_network(self, s, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            c0 = tf.layers.conv2d(s, 32, 8, 4, 'SAME', activation=tf.nn.relu, name='c0')  # [bs, 16, 16, 32]
            c1 = tf.layers.conv2d(c0, 64, 4, 2, 'SAME', activation=tf.nn.relu, name='c1')  # [bs, 8, 8, 64]
            c2 = tf.layers.conv2d(c1, 64, 3, 1, 'SAME', activation=tf.nn.relu, name='c2')  # [bs, 8, 8, 64]
            internal_rep = tf.reshape(c2, [-1, 8 * 8 * 64])
            fc1 = tf.layers.dense(internal_rep, 128, activation=tf.nn.relu, name='fc1')
            q = (1 / (1 - self.gamma)) * tf.layers.dense(fc1, self.num_actions, activation=tf.nn.sigmoid, name='q')
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.original_name_scope)
        return q, vars


    def setup_Q_functions(self):
        # build necessary q terms
        # variable update operations
        # soft_update_target = tf.group(*[tf.assign(target, tau * target + (1 - tau) * network)
        #                                 for network, target in zip(qa_vars, qa_target_vars)])
        # self.soft_update_target = soft_update_target
        # hard_update_target = tf.group(*[tf.assign(target, network)
        #                                 for network, target in zip(qa_vars, qa_target_vars)])
        Q_s, Q_sp = dict(), dict()
        soft_update_ops = []
        hard_update_ops = []
        tau = 0.998
        if self.use_shared_q_repr:
            builder_func = self.build_shared_Q_network_head
            inp_s, trunk_vars_s = self.build_shared_Q_network_trunk(self.converted_inp_s, 'Q_trunk')
            if self.use_target:
                inp_sp, trunk_vars_sp = self.build_shared_Q_network_trunk(self.converted_inp_sp, 'Q_trunk_target')
            else:
                inp_sp, trunk_vars_sp = self.build_shared_Q_network_trunk(self.converted_inp_sp, 'Q_trunk', reuse=True)
            hard_trunk_update = tf.group(*[tf.assign(target, network) for network, target in zip(trunk_vars_s, trunk_vars_sp)])
            soft_trunk_update = tf.group(*[tf.assign(target, tau * target + (1 - tau) * network)
                                           for network, target in zip(trunk_vars_s, trunk_vars_sp)])
            soft_update_ops.append(soft_trunk_update)
            hard_update_ops.append(hard_trunk_update)
        else:
            builder_func = self.build_Q_network
            inp_s = self.converted_inp_s
            inp_sp = self.converted_inp_sp

        for i in range(self.num_rewards):
            for j in range(self.num_rewards):
                Q_ij_s, Q_ij_s_vars = builder_func(inp_s, f'Q_{i}_{j}')
                Q_s[(i,j)] = Q_ij_s
                if self.use_target:
                    Q_ij_sp, Q_ij_sp_vars = builder_func(inp_sp, f'Q_{i}_{j}_target')
                else:
                    Q_ij_sp, Q_ij_sp_vars = builder_func(inp_sp, f'Q_{i}_{j}', reuse=True)

                Q_sp[(i,j)] = Q_ij_sp
                # build the update ops.
                hard_ij_update = tf.group(*[tf.assign(target, network) for network, target in zip(Q_ij_s_vars, Q_ij_sp_vars)])
                soft_ij_update = tf.group(*[tf.assign(target, tau * target + (1 - tau) * network)
                                            for network, target in zip(Q_ij_s_vars, Q_ij_sp_vars)])
                soft_update_ops.append(soft_ij_update)
                hard_update_ops.append(hard_ij_update)
        soft_update = tf.group(*soft_update_ops)
        hard_update = tf.group(*hard_update_ops)
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
        return Q_s, Q_sp, R, soft_update, hard_update


    def select_terms(self, term_list, num_terms_per_iter, num_elems):
        term_mask = tf.one_hot(tf.random_shuffle(tf.range(num_elems))[:num_terms_per_iter], num_elems) # [num_terms_per_iter, num_elems]
        return tf.reduce_sum(tf.reshape(term_list, [1, -1]) * term_mask, axis=1) # [num_terms_per_iter]

    def setup_constraints(self, Q_s, Q_sp, R):
        # set up reward_constraints
        if self.use_huber:
            loss = huber_loss
        else:
            loss = lambda x, y: tf.square(x - y)
        sums_to_R = tf.reduce_mean(loss(tf.reduce_sum([R[(i,i)] for i in range(self.num_rewards)], axis=0), self.inp_r), axis=0)
        greater_than_0 = tf.reduce_mean(tf.reduce_sum([tf.square(tf.maximum(0.0, -R[(i,i)])) for i in range(self.num_rewards)], axis=0), axis=0)

        # set up consistency constraints
        num_terms = 10
        reward_consistency_terms = []
        for i in range(self.num_rewards):
            for j in range(self.num_rewards):
                reward_consistency_terms.append(tf.reduce_mean(loss(R[(i,i)], R[(i,j)]), axis=0))
        selected_reward_consistency_terms = self.select_terms(reward_consistency_terms, num_terms, self.num_rewards**2)
        print('selected_reward_consistency_terms', selected_reward_consistency_terms)
        reward_consistency = tf.reduce_sum(selected_reward_consistency_terms, axis=0)


        # set up J_indep, J_nontriv
        #J_indep = 0
        #J_nontriv = 0
        J_nontriv_terms = []
        J_indep_terms = []
        for i in range(self.num_rewards):
            for j in range(self.num_rewards):
                if i == j:
                    V_ii = tf.reduce_max(Q_s[(i,j)], axis=1)
                    J_nontriv_terms.append(V_ii)
                else:
                    pi_j_action = tf.one_hot(tf.argmax(Q_s[(j,j)], axis=1), self.num_actions)
                    V_ij = tf.reduce_mean(tf.reduce_sum(Q_s[(i,j)] * pi_j_action, axis=1), axis=0)
                    J_indep_terms.append(V_ij)
        selected_J_indep_terms = self.select_terms(J_indep_terms, num_terms, self.num_rewards**2-self.num_rewards)
        print('selected_J_indep_terms', selected_J_indep_terms)
        J_indep = tf.reduce_sum(selected_J_indep_terms, axis=0)

        avg_max_values = tf.identity([tf.reduce_mean(J_nontriv_terms[i], axis=0) for i in range(self.num_rewards)])
        max_value_weighting = tf.stop_gradient(tf.nn.softmax(-2.0*avg_max_values))
        J_nontriv = 0
        for i in range(self.num_rewards):
            J_nontriv += tf.reduce_mean(J_nontriv_terms[i] * max_value_weighting[i], axis=0)
        return sums_to_R, greater_than_0, reward_consistency, J_indep, J_nontriv


