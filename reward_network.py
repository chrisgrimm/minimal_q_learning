import tensorflow as tf
import numpy as np
from q_learner_agent import QLearnerAgent

class RewardPartitionNetwork(object):

    def __init__(self, buffer, reward_buffer, num_partitions, obs_size, num_actions, name, visual=False, use_gpu=False, num_visual_channels=3, gpu_num=0, reuse=None):
        if not use_gpu:
            gpu_num = 0
        self.num_partitions = num_partitions
        self.num_actions = num_actions
        self.obs_size = obs_size
        self.buffer = buffer
        self.reward_buffer = reward_buffer
        self.visual = visual
        self.traj_len = 30
        self.num_visual_channels = num_visual_channels
        self.obs_shape = [None, self.obs_size] if not self.visual else [None, 64, 64, self.num_visual_channels]
        self.obs_shape_traj = [None, self.traj_len, self.obs_size] if not self.visual else [None, self.traj_len, 64, 64, self.num_visual_channels]


        self.Q_networks = [QLearnerAgent(obs_size, num_actions, f'qnet{i}', num_visual_channels=num_visual_channels, use_gpu=use_gpu, visual=visual, gpu_num=gpu_num)
                           for i in range(num_partitions)]

        with tf.device(f'/{"gpu" if use_gpu else "cpu"}:{gpu_num}'):
            with tf.variable_scope(name, reuse=reuse):
                if self.visual:
                    self.inp_sp = tf.placeholder(tf.uint8, self.obs_shape)
                    self.inp_sp_converted = tf.image.convert_image_dtype(self.inp_sp, dtype=tf.float32)
                else:
                    self.inp_sp = tf.placeholder(tf.float32, self.obs_shape)
                    self.inp_sp_converted = self.inp_sp
                self.inp_r = tf.placeholder(tf.float32, [None])
                #print('OG partitioned reward', self.inp_s, inp_a_onehot)
                partitioned_reward = self.partitioned_reward_tf(self.inp_sp_converted, 'reward_partition')
                self.partitioned_reward = partitioned_reward


                # build the list of placeholders
                self.list_inp_sp_traj = []
                self.list_inp_t_traj = []
                #self.list_inp_sp_traj_converted = []
                #self.list_reward_trajs = []
                self.list_trajectory_values = []
                for i in range(self.num_partitions):
                    if self.visual:
                        inp_sp_trajs_i_then_i = tf.placeholder(tf.uint8, self.obs_shape_traj)
                        self.list_inp_sp_traj.append(inp_sp_trajs_i_then_i)
                        inp_sp_trajs_i_then_i_converted = tf.image.convert_image_dtype(inp_sp_trajs_i_then_i,
                                                                                       dtype=tf.float32)
                    else:
                        inp_sp_trajs_i_then_i = tf.placeholder(tf.float32, self.obs_shape_traj)
                        self.list_inp_sp_traj.append(inp_sp_trajs_i_then_i)
                        inp_sp_trajs_i_then_i_converted = inp_sp_trajs_i_then_i

                    inp_t_trajs_i_then_i = tf.placeholder(tf.bool, [None, self.traj_len])
                    self.list_inp_t_traj.append(inp_t_trajs_i_then_i)

                    reward_trajs_i_then_i = self.partition_reward_traj(inp_sp_trajs_i_then_i_converted,
                                                                       name='reward_partition',
                                                                       reuse=True)
                    i_trajectory_values = self.get_values(reward_trajs_i_then_i, inp_t_trajs_i_then_i)
                    self.list_trajectory_values.append(i_trajectory_values)

                partition_constraint = 3*100*tf.reduce_mean(tf.square(self.inp_r - tf.reduce_sum(partitioned_reward, axis=1)))
                self.partition_loss = partition_constraint


                # build the value constraint
                value_constraint = 0
                for i in range(self.num_partitions):
                    for j in range(self.num_partitions):
                        if i == j:
                            continue
                        value_constraint += tf.square(self.list_trajectory_values[i][:, j])
                value_constraint = tf.reduce_mean(value_constraint, axis=0)
                self.value_loss = value_constraint

                self.loss = value_constraint + partition_constraint

                reward_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'{name}/reward_partition/')
                print(reward_params)
                self.train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(self.loss, var_list=reward_params)
                self.train_op_value = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(value_constraint)
                self.train_op_partition = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(partition_constraint)

            all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=f'{name}/')
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = sess = tf.Session(config=config)
            self.sess.run(tf.variables_initializer(all_variables))


    def train_Q_networks(self):
        Q_losses = []
        s_no_reward_batch, a_no_reward_batch, r_no_reward_batch, sp_no_reward_batch, t_no_reward_batch = self.buffer.sample(16)
        s_reward_batch, a_reward_batch, r_reward_batch, sp_reward_batch, t_reward_batch = self.reward_buffer.sample(16)
        s_batch = s_no_reward_batch + s_reward_batch
        a_batch = a_no_reward_batch + a_reward_batch
        r_batch = r_no_reward_batch + r_reward_batch
        sp_batch = sp_no_reward_batch + sp_reward_batch
        t_batch = t_no_reward_batch + t_reward_batch
        [partitioned_reward] = self.sess.run([self.partitioned_reward], feed_dict={self.inp_sp: sp_batch})
        for i, network in enumerate(self.Q_networks):
            loss = network.train_batch(s_batch, a_batch, partitioned_reward[:, i], sp_batch, t_batch)
            Q_losses.append(loss)
        return Q_losses

    def train_R_function_partition(self):
        batch_size = 32
        _, _, r_no_reward_batch, sp_no_reward_batch, t_batch = self.buffer.sample(batch_size // 2)
        _, _, r_reward_batch, sp_reward_batch, _ = self.reward_buffer.sample(batch_size // 2)
        r_batch = r_no_reward_batch + r_reward_batch
        sp_batch = sp_no_reward_batch + sp_reward_batch
        feed_dict = {
            self.inp_sp: sp_batch,
            self.inp_r: r_batch
        }
        [_, loss] = self.sess.run([self.train_op_partition, self.partition_loss], feed_dict=feed_dict)
        return loss

    def train_R_function_value(self, dummy_env_cluster):
        feed_dict = {}
        for j in range(self.num_partitions):
            dummy_env_cluster('reset', args=[])
            starting_states = [[x] for x in dummy_env_cluster('get_current_state', args=[])]
            SP_j_then_j, T_j_then_j = self.get_trajectory(dummy_env_cluster, starting_states, j, self.traj_len)
            feed_dict[self.list_inp_sp_traj[j]] = SP_j_then_j
            feed_dict[self.list_inp_t_traj[j]] = T_j_then_j
        [_, loss] = self.sess.run([self.train_op_value, self.value_loss], feed_dict=feed_dict)
        return loss




    def train_R_function(self, dummy_env_cluster):
        #batch_size = 32

        partition_loss = self.train_R_function_partition()
        value_loss = self.train_R_function_value(dummy_env_cluster)
        return partition_loss, value_loss
        # _, _, r_no_reward_batch, sp_no_reward_batch, t_batch = self.buffer.sample(batch_size // 2)
        # _, _, r_reward_batch, sp_reward_batch, _ = self.reward_buffer.sample(batch_size // 2)
        # r_batch = r_no_reward_batch + r_reward_batch
        # sp_batch = sp_no_reward_batch + sp_reward_batch
        #
        # # collect  all the trajectories.
        # all_SP_traj_batches = []
        # all_T_traj_batches = []
        #
        #     # initialize the environment randomly and collect the initial state. this allows us to perform the necessary
        #     # resets to estimate values.
        # #dummy_env_cluster('reset', args=[])
        # #starting_state = dummy_env.get_current_state()
        # #starting_states = dummy_env_cluster('get_current_state', args=[])
        #
        # feed_dict = {
        #     self.inp_sp: sp_batch,
        #     self.inp_r: r_batch
        # }
        #
        # for j in range(self.num_partitions):
        #     dummy_env_cluster('reset', args=[])
        #     starting_states = [[x] for x in dummy_env_cluster('get_current_state', args=[])]
        #     SP_j_then_j, T_j_then_j = self.get_trajectory(dummy_env_cluster, starting_states, j, self.traj_len)
        #     feed_dict[self.list_inp_sp_traj[j]] = SP_j_then_j
        #     feed_dict[self.list_inp_t_traj[j]] = T_j_then_j
        #
        #
        #
        # # for i in range(self.num_partitions):
        # #     feed_dict[self.list_inp_sp_traj[i]] = all_SP_traj_batches[i]
        # #     feed_dict[self.list_inp_t_traj[i]] = all_T_traj_batches[i]
        #
        # [_, loss] = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        # return loss



    # grab sample trajectories from a starting state.
    def get_trajectory(self, dummy_env_cluster, starting_states, policy, trajectory_length):
        sp_traj = []
        t_traj = []
        #s0 = dummy_env.restore_state(starting_state)
        s_list = dummy_env_cluster('restore_state', sharded_args=starting_states)
        for i in range(trajectory_length):
            #a = self.Q_networks[policy].get_action([s0])[0]
            a_list = [[x] for x in self.Q_networks[policy].get_action(s_list)]
            #s, _, t, _ = dummy_env.step(a)
            #(sp, r, t, info)
            experience_tuple_list = dummy_env_cluster('step', sharded_args=a_list)
            s_list = [sp for (sp, _, t, _) in experience_tuple_list]
            sp_traj.append([sp for (sp, _, t, _) in experience_tuple_list])
            t_traj.append([t for (sp, _, t, _) in experience_tuple_list])
        sp_traj = np.transpose(sp_traj, [1, 0, 2, 3, 4])
        t_traj = np.transpose(t_traj, [1, 0])
        return sp_traj, t_traj


    def partitioned_reward_tf(self, sp, name, reuse=None):
        if self.visual:
            return self.partitioned_reward_tf_visual(sp, name, reuse)
        else:
            return self.partitioned_reward_tf_vector(sp, name, reuse)


    def partitioned_reward_tf_vector(self, sp, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            #inp = tf.concat([s, a], axis=1)
            fc1 = tf.layers.dense(sp, 128, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, name='fc2')
            rewards = tf.layers.dense(fc2, len(self.Q_networks), activation=tf.nn.sigmoid, name='rewards')
        return rewards


    def partitioned_reward_tf_visual(self, sp, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            # sp : [bs, 32, 32 ,3]
            x = sp
            x = tf.layers.conv2d(x, 32, 3, 2, 'SAME', activation=tf.nn.relu, name='c0') # [bs, 32, 32, 32]
            x = tf.layers.conv2d(x, 32, 3, 2, 'SAME', activation=tf.nn.relu, name='c1')  # [bs, 16, 16, 32]
            x = tf.layers.conv2d(x, 32, 3, 2, 'SAME', activation=tf.nn.relu, name='c2')  # [bs, 8, 8, 32]
            x = tf.layers.dense(tf.reshape(x, [-1, 8 * 8 * 32]), 128, activation=tf.nn.relu, name='fc1')
            rewards = tf.layers.dense(x, len(self.Q_networks), activation=tf.nn.sigmoid, name='qa')
        return rewards

    def partition_reward_traj(self, sp_traj, name, reuse=None):
        print(sp_traj)
        Rs_traj = []
        for t in range(self.traj_len):
            #s = s_traj[:, t, :]
            #a = a_traj[:, t, :]
            sp = sp_traj[:, t, :]
            Rs = self.partitioned_reward_tf(sp, name, reuse=(t > 0) or (reuse == True))
            Rs_traj.append(tf.reshape(Rs, [-1, 1, self.num_partitions])) # [bs, 1, n]
        return tf.concat(Rs_traj, axis=1) # [bs, traj, n]


    def get_partitioned_reward(self, sp):
        [r]  = self.sess.run([self.partitioned_reward], feed_dict={self.inp_sp: sp})
        return r

    def get_state_values(self, s):
        return [np.max(self.Q_networks[i].get_Q(s), axis=1) for i in range(self.num_partitions)]

    def get_state_actions(self, s):
        return [self.Q_networks[i].get_action(s) for i in range(self.num_partitions)]

    #def get_state_rewards(self, s):
    #    return self.get_partitioned_reward([s]*5, list(range(5)))

    def get_reward(self, sp):
        return self.get_partitioned_reward([sp])[0]


    # returns vector V where V_i = value of trajectory under reward i.
    def get_values(self, rs_traj, ts_traj):
        # rs_traj : [bs, traj_len, num_partitions]
        # ts_traj : [bs, traj_len]
        print(rs_traj)
        gamma = 0.99
        gamma_sequence = tf.reshape(tf.pow(gamma, list(range(self.traj_len))), [1, self.traj_len, 1])
        t_sequence = 1.0 - tf.reshape(tf.cast(ts_traj, tf.float32), [-1, self.traj_len, 1])
        # after the first terminal state, sticky_t_sequence should always be 0.
        sticky_t_sequence = tf.cumprod(t_sequence, axis=1)
        #prod_reward = 0.0
        out = tf.reduce_sum(rs_traj * gamma_sequence * sticky_t_sequence, axis=1) # [bs, num_partitions]
        print('out', out)
        return out

