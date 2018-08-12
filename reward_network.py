import tensorflow as tf
import numpy as np
from q_learner_agent import QLearnerAgent

class RewardPartitionNetwork(object):

    def __init__(self, buffer, goal_buffer, num_partitions, obs_size, num_actions, name, reuse=None):
        self.num_partitions = num_partitions
        self.num_actions = num_actions
        self.obs_size = obs_size
        self.buffer = buffer
        self.goal_buffer = goal_buffer

        self.traj_len = 10


        self.Q_networks = [QLearnerAgent(obs_size, num_actions, f'qnet{i}')
                           for i in range(num_partitions)]

        with tf.variable_scope(name, reuse=reuse):

            self.inp_s = tf.placeholder(tf.float32, [None, self.obs_size])
            self.inp_a = tf.placeholder(tf.int32, [None])
            inp_a_onehot = tf.one_hot(self.inp_a, self.num_actions)
            self.inp_r = tf.placeholder(tf.float32, [None])
            print('OG partitioned reward', self.inp_s, inp_a_onehot)
            partitioned_reward = self.partitioned_reward_tf(self.inp_s, inp_a_onehot, 'reward_partition')
            self.partitioned_reward = partitioned_reward

            ################
            # pi1_then_pi2
            self.inp_s_trajs_pi1_then_pi2 = tf.placeholder(tf.float32, [None, self.traj_len, self.obs_size])
            self.inp_a_trajs_pi1_then_pi2 = tf.placeholder(tf.int32, [None, self.traj_len])

            self.a_onehot_trajs_pi1_then_pi2 = tf.one_hot(self.inp_a_trajs_pi1_then_pi2, self.num_actions) # [bs, traj_len, num_actions]
            self.reward_trajs_pi1_then_pi2 = self.partition_reward_traj(self.inp_s_trajs_pi1_then_pi2, self.a_onehot_trajs_pi1_then_pi2, name='reward_partition', reuse=True) #[bs, num_partitions, traj_len]
            pi1_then_pi2_values, _ = self.get_values(self.reward_trajs_pi1_then_pi2)
            Q2_pi1 = pi1_then_pi2_values[:,1] # [num_partitions]

            ################
            # pi1_then_pi1
            self.inp_s_trajs_pi1_then_pi1 = tf.placeholder(tf.float32, [None, self.traj_len, self.obs_size])
            self.inp_a_trajs_pi1_then_pi1 = tf.placeholder(tf.int32, [None, self.traj_len])
            self.a_onehot_trajs_pi1_then_pi1 = tf.one_hot(self.inp_a_trajs_pi1_then_pi1, self.num_actions)  # [bs, traj_len, num_actions]
            self.reward_trajs_pi1_then_pi1 = self.partition_reward_traj(self.inp_s_trajs_pi1_then_pi1,
                                                                        self.a_onehot_trajs_pi1_then_pi1,
                                                                        name='reward_partition', reuse=True)  # [bs, traj_len, num_partitions]
            pi1_then_pi1_values, pi1_then_pi1_prod_values = self.get_values(self.reward_trajs_pi1_then_pi1)
            Q1_pi1 = pi1_then_pi1_values[:,0]  # [num_partitions]
            value2_on_policy1 = pi1_then_pi1_values[:, 1]
            prod_value_on_policy1 = pi1_then_pi1_prod_values[:, 0]
            first_reward2_on_policy1 = self.reward_trajs_pi1_then_pi1[:, 0, 1]

            #################
            # pi2_then_pi1
            self.inp_s_trajs_pi2_then_pi1 = tf.placeholder(tf.float32, [None, self.traj_len, self.obs_size])
            self.inp_a_trajs_pi2_then_pi1 = tf.placeholder(tf.int32, [None, self.traj_len])
            self.a_onehot_trajs_pi2_then_pi1 = tf.one_hot(self.inp_a_trajs_pi2_then_pi1, self.num_actions) # [bs, traj_len, num_actions]
            self.reward_trajs_pi2_then_pi1 = self.partition_reward_traj(self.inp_s_trajs_pi2_then_pi1,
                                                                        self.a_onehot_trajs_pi2_then_pi1,
                                                                        name='reward_partition', reuse=True)  # [bs, traj_len, num_partitions]
            pi2_then_pi1_values, _ = self.get_values(self.reward_trajs_pi2_then_pi1)
            Q1_pi2 = pi2_then_pi1_values[:,0]  # [num_partitions]

            #################
            # pi2_then_pi2
            self.inp_s_trajs_pi2_then_pi2 = tf.placeholder(tf.float32, [None, self.traj_len, self.obs_size])
            self.inp_a_trajs_pi2_then_pi2 = tf.placeholder(tf.int32, [None, self.traj_len])
            self.a_onehot_trajs_pi2_then_pi2 = tf.one_hot(self.inp_a_trajs_pi2_then_pi2, self.num_actions)  # [bs, traj_len, num_actions]
            self.reward_trajs_pi2_then_pi2 = self.partition_reward_traj(self.inp_s_trajs_pi2_then_pi2,
                                                                        self.a_onehot_trajs_pi2_then_pi2,
                                                                        name='reward_partition', reuse=True)  # [bs, traj_len, num_partitions]

            pi2_then_pi2_values, pi2_then_pi2_prod_values = self.get_values(self.reward_trajs_pi2_then_pi2)
            Q2_pi2 = pi2_then_pi2_values[:,1]  # [num_partitions]
            value1_on_policy2 = pi2_then_pi2_values[:, 0]
            prod_value_on_policy2 = pi2_then_pi2_prod_values[:, 0]
            first_reward1_on_policy2 = self.reward_trajs_pi2_then_pi2[:, 0, 0]






            partition_constraint = 100*tf.reduce_mean(tf.square(self.inp_r - tf.reduce_prod(partitioned_reward, axis=1)))
            Q_constraint = tf.reduce_mean(tf.square(value1_on_policy2) +
                                          tf.square(value2_on_policy1) +
                                          (-prod_value_on_policy2 * tf.stop_gradient(first_reward1_on_policy2)) +
                                          (-prod_value_on_policy1 * tf.stop_gradient(first_reward2_on_policy1))
                                          )
            self.loss = Q_constraint + partition_constraint

            reward_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'{name}/reward_partition/')
            print(reward_params)
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(self.loss, var_list=reward_params)

        all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=f'{name}/')
        self.sess = tf.Session()
        self.sess.run(tf.variables_initializer(all_variables))


    def train_Q_networks(self):
        Q_losses = []
        s_batch, a_batch, r_batch, sp_batch, t_batch = self.buffer.sample(32)
        [partitioned_reward] = self.sess.run([self.partitioned_reward], feed_dict={self.inp_s: s_batch, self.inp_a: a_batch})
        for i, network in enumerate(self.Q_networks):
            loss = network.train_batch(s_batch, a_batch, partitioned_reward[:, i], sp_batch, t_batch)
            Q_losses.append(loss)
        return Q_losses



    def train_R_function(self, dummy_env):
        batch_size = 32
        s_batch_norm, a_batch_norm, r_batch_norm, _, _ = self.buffer.sample(batch_size // 2)
        s_batch_goal, a_batch_goal, r_batch_goal, _, _ = self.goal_buffer.sample(batch_size // 2)
        s_batch = s_batch_norm + s_batch_goal
        a_batch = a_batch_norm + a_batch_goal
        r_batch = r_batch_norm + r_batch_goal


        # collect  all the trajectories.
        S_pi1_then_pi2_batch, A_pi1_then_pi2_batch = [], []
        S_pi1_then_pi1_batch, A_pi1_then_pi1_batch = [], []
        S_pi2_then_pi1_batch, A_pi2_then_pi1_batch = [], []
        S_pi2_then_pi2_batch, A_pi2_then_pi2_batch = [], []

        for i in range(batch_size):
            dummy_env.reset()
            starting_state = dummy_env.get_current_state()

            S_pi1_then_pi2, A_pi1_then_pi2 = self.get_trajectory(dummy_env, starting_state, 0, 1, self.traj_len)
            S_pi1_then_pi2_batch.append(S_pi1_then_pi2)
            A_pi1_then_pi2_batch.append(A_pi1_then_pi2)

            S_pi1_then_pi1, A_pi1_then_pi1 = self.get_trajectory(dummy_env, starting_state, 0, 0, self.traj_len)
            S_pi1_then_pi1_batch.append(S_pi1_then_pi1)
            A_pi1_then_pi1_batch.append(A_pi1_then_pi1)

            S_pi2_then_pi1, A_pi2_then_pi1 = self.get_trajectory(dummy_env, starting_state, 1, 0, self.traj_len)
            S_pi2_then_pi1_batch.append(S_pi2_then_pi1)
            A_pi2_then_pi1_batch.append(A_pi2_then_pi1)

            S_pi2_then_pi2, A_pi2_then_pi2 = self.get_trajectory(dummy_env, starting_state, 1, 1, self.traj_len)
            S_pi2_then_pi2_batch.append(S_pi2_then_pi2)
            A_pi2_then_pi2_batch.append(A_pi2_then_pi2)


        [_, loss] = self.sess.run([self.train_op, self.loss], feed_dict={
            self.inp_s: s_batch,
            self.inp_a: a_batch,
            self.inp_r: r_batch,

            self.inp_s_trajs_pi1_then_pi2: S_pi1_then_pi2_batch,
            self.inp_a_trajs_pi1_then_pi2: A_pi1_then_pi2_batch,

            self.inp_s_trajs_pi1_then_pi1: S_pi1_then_pi1_batch,
            self.inp_a_trajs_pi1_then_pi1: A_pi1_then_pi1_batch,

            self.inp_s_trajs_pi2_then_pi1: S_pi2_then_pi1_batch,
            self.inp_a_trajs_pi2_then_pi1: A_pi2_then_pi1_batch,

            self.inp_s_trajs_pi2_then_pi2: S_pi2_then_pi2_batch,
            self.inp_a_trajs_pi2_then_pi2: A_pi2_then_pi2_batch
        })

        return loss



    # grab sample trajectories from a starting state.
    def get_trajectory(self, dummy_env, starting_state, starting_policy, policy_thereafter, trajectory_length):
        s_traj = []
        a_traj = []
        s0 = dummy_env.restore_state(starting_state)
        a0 = self.Q_networks[starting_policy].get_action([s0])[0]
        s_traj.append(s0)
        a_traj.append(a0)
        a = a0
        for i in range(trajectory_length-1):
            s, _, _, _ = dummy_env.step(a)
            a = self.Q_networks[policy_thereafter].get_action([s])[0]
            s_traj.append(s)
            a_traj.append(a)
        return s_traj, a_traj




    def partitioned_reward_tf(self, s, a, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            inp = tf.concat([s, a], axis=1)
            fc1 = tf.layers.dense(inp, 64, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 64, activation=tf.nn.relu, name='fc2')
            rewards = tf.layers.dense(fc2, len(self.Q_networks), activation=tf.nn.sigmoid, name='rewards')
        return rewards

    def partition_reward_traj(self, s_traj, a_traj, name, reuse=None):
        print(s_traj)
        Rs_traj = []
        for t in range(self.traj_len):
            s = s_traj[:, t, :]
            a = a_traj[:, t, :]
            Rs = self.partitioned_reward_tf(s, a, name, reuse=(t > 0) or (reuse == True))
            Rs_traj.append(tf.reshape(Rs, [-1, 1, self.num_partitions])) # [bs, 1, n]
        return tf.concat(Rs_traj, axis=1) # [bs, traj, n]


    def get_partitioned_reward(self, s, a):
        [r]  = self.sess.run([self.partitioned_reward], feed_dict={self.inp_s: s, self.inp_a: a})
        return r

    def get_state_values(self, s):
        return [np.max(self.Q_networks[i].get_Q(s), axis=1) for i in range(self.num_partitions)]

    def get_state_actions(self, s):
        processed_s = []
        none_indices = []
        for i in range(len(s)):
            if s[i] is None:
                processed_s.append(np.zeros([self.obs_size]))
                none_indices.append(i)
            else:
                processed_s.append(s[i])

        return [self.Q_networks[i].get_action(processed_s) if i not in none_indices else None
                for i in range(self.num_partitions)]

    def get_state_rewards(self, s):
        return self.get_partitioned_reward([s]*5, list(range(5)))


    def get_values(self, rs_traj):
        # rs_traj : [bs, traj_len, num_partitions]
        print(rs_traj)
        gamma = 0.99
        gamma_sequence = tf.reshape(tf.pow(gamma, list(range(self.traj_len))), [1, self.traj_len, 1])
        prod_reward = tf.reduce_prod(rs_traj, axis=2, keep_dims=True) # [ bs, traj_len, 1]
        #prod_reward = 0.0
        out = tf.reduce_sum(rs_traj * (1.0 - prod_reward) * gamma_sequence, axis=1) # [bs, num_partitions]
        out_prod = tf.reduce_sum(prod_reward * gamma_sequence, axis=1)  # [bs, 1]
        print('out', out)
        return out, out_prod

