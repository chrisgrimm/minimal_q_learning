import tensorflow as tf
import numpy as np
from q_learner_agent import QLearnerAgent

class RewardPartitionNetwork(object):

    def __init__(self, buffer, reward_buffer, num_partitions, obs_size, num_actions, name, traj_len=30,
                 max_value_mult=10, use_dynamic_weighting_max_value=True, use_dynamic_weighting_disentangle_value=False,
                 visual=False, num_visual_channels=3, gpu_num=0, use_gpu=False, lr=0.0001, reuse=None):
        if not use_gpu:
            gpu_num = 0
        self.num_partitions = num_partitions
        self.num_actions = num_actions
        self.obs_size = obs_size
        self.buffer = buffer
        self.reward_buffer = reward_buffer
        self.visual = visual
        self.traj_len = traj_len
        self.max_value_mult = max_value_mult
        self.use_dynamic_weighting_max_value = use_dynamic_weighting_max_value
        self.use_dynamic_weighting_disentangle_value = use_dynamic_weighting_disentangle_value

        self.num_visual_channels = num_visual_channels
        self.obs_shape = [None, self.obs_size] if not self.visual else [None, 64, 64, self.num_visual_channels]
        self.obs_shape_traj = [None, self.traj_len, self.obs_size] if not self.visual else [None, self.traj_len, 64, 64, self.num_visual_channels]


        self.Q_networks = [QLearnerAgent(obs_size, num_actions, f'qnet{i}', num_visual_channels=num_visual_channels, visual=visual, gpu_num=gpu_num, use_gpu=use_gpu)
                           for i in range(num_partitions)]
        with tf.device(f'/{"gpu" if use_gpu else "cpu"}:{gpu_num}'):
            with tf.variable_scope(name, reuse=reuse):
                #self.inp_only_rewarding_trajectories = tf.placeholder(tf.bool)
                if self.visual:
                    self.inp_sp = tf.placeholder(tf.uint8, self.obs_shape)
                    self.inp_sp_converted = tf.image.convert_image_dtype(self.inp_sp, dtype=tf.float32)
                else:
                    self.inp_sp = tf.placeholder(tf.float32, self.obs_shape)
                    self.inp_sp_converted = self.inp_sp
                self.inp_r = tf.placeholder(tf.float32, [None])
                #print('OG partitioned reward', self.inp_s, inp_a_onehot)
                partitioned_reward = self.partitioned_reward_tf(self.inp_sp_converted, self.inp_r, 'reward_partition')
                self.partitioned_reward = partitioned_reward


                # build the list of placeholders
                self.list_inp_sp_traj = []
                self.list_inp_r_traj = []
                self.list_inp_t_traj = []
                #self.list_inp_sp_traj_converted = []
                #self.list_reward_trajs = []
                self.list_trajectory_values = []
                #self.list_any_r = []
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

                    inp_r_trajs_i_then_i = tf.placeholder(tf.float32, [None, self.traj_len])
                    inp_t_trajs_i_then_i = tf.placeholder(tf.bool, [None, self.traj_len])
                    self.list_inp_r_traj.append(inp_r_trajs_i_then_i)
                    self.list_inp_t_traj.append(inp_t_trajs_i_then_i)

                    any_rewards = tf.reduce_any(tf.equal(inp_r_trajs_i_then_i, 1), axis=1)
                    #any_rewards = tf.cast(any_rewards, tf.float32)
                    #self.list_any_r.append(any_rewards)
                    reward_trajs_i_then_i = self.partition_reward_traj(inp_sp_trajs_i_then_i_converted,
                                                                       inp_r_trajs_i_then_i,
                                                                       name='reward_partition',
                                                                       reuse=True)
                    i_trajectory_values = self.get_values(reward_trajs_i_then_i, inp_t_trajs_i_then_i)
                    self.list_trajectory_values.append(i_trajectory_values)

                #partition_constraint = 3*100*tf.reduce_mean(tf.square(self.inp_r - tf.reduce_sum(partitioned_reward, axis=1)))


                if self.use_dynamic_weighting_max_value:
                    avg_max_values = tf.identity(
                        [tf.reduce_mean(self.list_trajectory_values[i][:, i], axis=0) for i in range(self.num_partitions)])
                    max_value_weighting = tf.stop_gradient(tf.nn.softmax(-avg_max_values))  # [num_partitions]
                else:
                    max_value_weighting = tf.ones(shape=[self.num_partitions], dtype=tf.float32)


                max_value_constraint = 0
                for i in range(self.num_partitions):
                    max_value_constraint += max_value_weighting[i] * self.list_trajectory_values[i][:, i]
                max_value_constraint = tf.reduce_mean(max_value_constraint, axis=0)
                #max_value_constraint = tf.reduce_mean(
                #    tf.reduce_min([self.list_trajectory_values[i][:, i] for i in range(self.num_partitions)], axis=0))


                if self.use_dynamic_weighting_disentangle_value:
                    ordered = []
                    for i in range(self.num_partitions):
                        for j in range(self.num_partitions):
                            if i == j:
                                continue
                            ordered.append(tf.reduce_mean(self.list_trajectory_values[i][:, j], axis=0))
                    # No negative in front of this term because we want it to be small.
                    dist_value_weighting = tf.stop_gradient(tf.nn.softmax(tf.identity(ordered)))
                else:
                    dist_value_weighting = tf.ones(shape=[self.num_partitions**2 - self.num_partitions], dtype=tf.float32)

                #build the value constraint
                index = 0
                value_constraint = 0
                for i in range(self.num_partitions):
                   for j in range(self.num_partitions):
                       if i == j:
                           continue
                       value_constraint += (dist_value_weighting[index] * self.list_trajectory_values[i][:, j])
                       index += 1

                value_constraint = tf.reduce_mean(value_constraint, axis=0)



                self.max_value_constraint = max_value_constraint
                self.value_constraint = value_constraint
                self.loss = (value_constraint - self.max_value_mult*max_value_constraint)

                reward_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'{name}/reward_partition/')
                print(reward_params)
                self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, var_list=reward_params)

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
        [partitioned_reward] = self.sess.run([self.partitioned_reward], feed_dict={self.inp_sp: sp_batch, self.inp_r: r_batch})
        for i, network in enumerate(self.Q_networks):
            loss = network.train_batch(s_batch, a_batch, partitioned_reward[:, i], sp_batch, t_batch)
            Q_losses.append(loss)
        return Q_losses



    def train_R_function(self, dummy_env_cluster):
        batch_size = 32

        # _, _, r_no_reward_batch, sp_no_reward_batch, t_batch = self.buffer.sample(batch_size // 2)
        # _, _, r_reward_batch, sp_reward_batch, _ = self.reward_buffer.sample(batch_size // 2)
        # r_batch = r_no_reward_batch + r_reward_batch
        # sp_batch = sp_no_reward_batch + sp_reward_batch

        # collect  all the trajectories.
        #all_SP_traj_batches = []
        #all_T_traj_batches = []

            # initialize the environment randomly and collect the initial state. this allows us to perform the necessary
            # resets to estimate values.
        #dummy_env_cluster('reset', args=[])
        #starting_state = dummy_env.get_current_state()
        #starting_states = dummy_env_cluster('get_current_state', args=[])

        feed_dict = {}

        for j in range(self.num_partitions):
            dummy_env_cluster('reset', args=[])
            starting_states = [[x] for x in dummy_env_cluster('get_current_state', args=[])]
            SP_j_then_j, R_j_then_j, T_j_then_j = self.get_trajectory(dummy_env_cluster, starting_states, j, self.traj_len)
            feed_dict[self.list_inp_sp_traj[j]] = SP_j_then_j
            feed_dict[self.list_inp_r_traj[j]] = R_j_then_j
            feed_dict[self.list_inp_t_traj[j]] = T_j_then_j



        # for i in range(self.num_partitions):
        #     feed_dict[self.list_inp_sp_traj[i]] = all_SP_traj_batches[i]
        #     feed_dict[self.list_inp_t_traj[i]] = all_T_traj_batches[i]
        [_, loss, max_value_constraint, value_constraint] = self.sess.run([self.train_op, self.loss, self.max_value_constraint, self.value_constraint], feed_dict=feed_dict)
        return loss, max_value_constraint, value_constraint



    def get_action_stoch(self, policy, s_list, rand_prob=0.1):
        U = np.random.uniform(0,1, size=len(s_list))
        rand_a_list = np.random.randint(0, self.num_actions, size=len(s_list))
        a_list = self.Q_networks[policy].get_action(s_list)
        # TODO double check this line.
        mix_rand_a_list = rand_a_list * (U < rand_prob).astype(np.float32) + a_list * (U >= rand_prob).astype(np.float32)
        return mix_rand_a_list

    # grab sample trajectories from a starting state.
    def get_trajectory(self, dummy_env_cluster, starting_states, policy, trajectory_length):
        sp_traj = []
        t_traj = []
        r_traj = []
        #s0 = dummy_env.restore_state(starting_state)
        s_list = dummy_env_cluster('restore_state', sharded_args=starting_states)
        for i in range(trajectory_length):
            #a = self.Q_networks[policy].get_action([s0])[0]
            a_list = [[x] for x in self.Q_networks[policy].get_action(s_list)]
            #s, _, t, _ = dummy_env.step(a)
            #(sp, r, t, info)
            experience_tuple_list = dummy_env_cluster('step', sharded_args=a_list)
            # THIS IS NECESSARY, YOU DIPSHIT. the s_list needs to be updated so the actions make sense.
            s_list = [sp for (sp, r, t, _) in experience_tuple_list]

            r_traj.append([r for (sp, r, t, _) in experience_tuple_list])
            sp_traj.append([sp for (sp, r, t, _) in experience_tuple_list])
            t_traj.append([t for (sp, r, t, _) in experience_tuple_list])
        sp_traj = np.transpose(sp_traj, [1, 0, 2, 3, 4])
        t_traj = np.transpose(t_traj, [1, 0])
        r_traj = np.transpose(r_traj, [1, 0])
        return sp_traj, r_traj, t_traj


    def partitioned_reward_tf(self, sp, r, name, reuse=None):
        if self.visual:
            return self.partitioned_reward_tf_visual(sp, r, name, reuse)
        else:
            return self.partitioned_reward_tf_vector(sp, r, name, reuse)


    def partitioned_reward_tf_vector(self, sp, r, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            #inp = tf.concat([s, a], axis=1)
            fc1 = tf.layers.dense(sp, 128, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, name='fc2')
            soft = tf.nn.softmax(tf.layers.dense(fc2, len(self.Q_networks), name='rewards')) # [bs, num_partitions]
            rewards = tf.reshape(r, [-1, 1]) * soft
        return rewards


    def partitioned_reward_tf_visual(self, sp, r, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            # sp : [bs, 32, 32 ,3]
            print('r', r)
            x = sp
            x = tf.layers.conv2d(x, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c0') # [bs, 32, 32, 32]
            x = tf.layers.conv2d(x, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c1')  # [bs, 16, 16, 32]
            x = tf.layers.conv2d(x, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c2')  # [bs, 8, 8, 32]
            x = tf.layers.dense(tf.reshape(x, [-1, 8 * 8 * 32]), 128, activation=tf.nn.relu, name='fc1')
            soft = tf.layers.dense(x, len(self.Q_networks), activation=tf.nn.softmax, name='qa')
            #error_control = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name='error_control') # [bs, 1]

            rewards = tf.reshape(r, [-1, 1]) * soft #* error_control
        return rewards

    def partition_reward_traj(self, sp_traj, r_traj, name, reuse=None):
        Rs_traj = []
        for t in range(self.traj_len):
            #s = s_traj[:, t, :]
            #a = a_traj[:, t, :]
            sp = sp_traj[:, t, :]
            r = r_traj[:, t]
            print('r', r)
            Rs = self.partitioned_reward_tf(sp, r, name, reuse=(t > 0) or (reuse == True))
            Rs_traj.append(tf.reshape(Rs, [-1, 1, self.num_partitions])) # [bs, 1, n]
        return tf.concat(Rs_traj, axis=1) # [bs, traj, n]


    def get_partitioned_reward(self, sp, r):
        [partitioned_r]  = self.sess.run([self.partitioned_reward], feed_dict={self.inp_sp: sp, self.inp_r: r})
        return partitioned_r

    def get_state_values(self, s):
        return [np.max(self.Q_networks[i].get_Q(s), axis=1) for i in range(self.num_partitions)]

    def get_state_actions(self, s):
        return [self.Q_networks[i].get_action(s) for i in range(self.num_partitions)]

    #def get_state_rewards(self, s):
    #    return self.get_partitioned_reward([s]*5, list(range(5)))

    def get_reward(self, sp, r):
        return self.get_partitioned_reward([sp], [r])[0]


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

