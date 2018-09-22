import tensorflow as tf
import numpy as np
import cv2
import tqdm
from q_learner_agent import QLearnerAgent

class RewardPartitionNetwork(object):

    def __init__(self, buffer, reward_buffer, num_partitions, obs_size, num_actions, name, traj_len=30,
                 max_value_mult=10, use_dynamic_weighting_max_value=True, use_dynamic_weighting_disentangle_value=False,
                 visual=False, num_visual_channels=3, gpu_num=0, use_gpu=False, lr=0.0001, reuse=None, reuse_visual_scoping=False,
                 separate_reward_repr=False, use_ideal_threshold=False, reward_mode='SUM'):
        assert not (separate_reward_repr and reuse_visual_scoping)
        assert reward_mode in ['SUM', 'PROD']
        self.reward_mode = reward_mode
        if not use_gpu:
            gpu_num = 0
        self.threshold = np.ones(shape=[64, 64, 1], dtype=np.uint8)
        self.use_ideal_threshold = use_ideal_threshold
        if use_ideal_threshold:
            self.ideal_threshold = (cv2.imread('./ideal_threshold.png')[:, :, [0]] / 255).astype(np.uint8)
        else:
            self.ideal_threshold = None

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
        self.reuse_visual_scoping = reuse_visual_scoping
        self.separate_reward_repr = separate_reward_repr

        self.num_visual_channels = num_visual_channels
        self.obs_shape = [None, self.obs_size] if not self.visual else [None, 64, 64, self.num_visual_channels]
        self.obs_shape_traj = [None, self.traj_len, self.obs_size] if not self.visual else [None, self.traj_len, 64, 64, self.num_visual_channels]
        self.visual_scope = None
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)

        if reuse_visual_scoping:
            self.Q_networks = []
            qnet0 = QLearnerAgent(obs_size, num_actions, f'qnet0', num_visual_channels=num_visual_channels,
                                  visual=visual, gpu_num=gpu_num, use_gpu=use_gpu, sess=self.sess)
            self.Q_networks.append(qnet0)
            self.visual_scope = qnet0.visual_scope
            for i in range(1, num_partitions):
                self.Q_networks.append(
                    QLearnerAgent(obs_size, num_actions, f'qnet{i}', num_visual_channels=num_visual_channels,
                                  visual=visual, gpu_num=gpu_num, use_gpu=use_gpu,
                                  alternate_visual_scope=self.visual_scope, sess=self.sess)
                )
        else:
            self.Q_networks = [QLearnerAgent(obs_size, num_actions, f'qnet{i}', num_visual_channels=num_visual_channels,
                                             visual=visual, gpu_num=gpu_num, use_gpu=use_gpu, sess=self.sess)
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
                self.pred_reward, self.internal_repr, self.all_layers = self.reward_visual_tf(self.inp_sp_converted, 'pred_reward')



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
                self.list_traj_start_rewards = []
                self.list_diff = []
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
                    diff_i = tf.abs(reward_trajs_i_then_i[:, 1:self.traj_len, :] - reward_trajs_i_then_i[:, 0:self.traj_len-1, :])
                    diff_i = tf.reduce_mean(diff_i, axis=1) # [bs, num_partitions]
                    self.list_diff.append(diff_i)
                    self.list_traj_start_rewards.append(reward_trajs_i_then_i[:, 0, :]) # [bs, num_partitions]
                    i_trajectory_values = self.get_values(reward_trajs_i_then_i, inp_t_trajs_i_then_i)
                    self.list_trajectory_values.append(i_trajectory_values)

                self.inp_sp_mixed_traj = tf.placeholder(tf.uint8, self.obs_shape_traj)
                inp_sp_mixed_traj_converted = tf.image.convert_image_dtype(self.inp_sp_mixed_traj, dtype=tf.float32)
                self.inp_r_mixed_traj = tf.placeholder(tf.float32, [None, self.traj_len])
                self.inp_t_mixed_traj = tf.placeholder(tf.bool, [None, self.traj_len])

                prod_reward_traj_1 = self.partition_reward_traj(tf.image.convert_image_dtype(self.list_inp_sp_traj[1], tf.float32),
                                                                self.list_inp_r_traj[1],
                                                                name='reward_partition',
                                                                reuse=True)
                prod_reward_traj_1 = tf.reduce_prod(prod_reward_traj_1, axis=2, keep_dims=True) #[bs, traj_len]
                # TODO fix hack
                value_prod_reward_1 = self.get_values(prod_reward_traj_1, self.inp_t_mixed_traj)

                reward_trajs_mixed = self.partition_reward_traj(inp_sp_mixed_traj_converted, self.inp_r_mixed_traj, name='reward_partition', reuse=True)
                prod_reward_traj = tf.reduce_prod(reward_trajs_mixed, axis=2, keep_dims=True) #[bs, traj_len, 1]
                mixed_values = self.get_values(prod_reward_traj, self.inp_t_mixed_traj) #[bs, 1]
                print(mixed_values)






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


                prod_diff_constraint = 0
                for i in range(self.num_partitions):
                    for j in range(self.num_partitions):
                        if i == j:
                            continue
                        prod_diff_constraint += self.list_diff[i][:, j]




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

                self.reward_loss = tf.reduce_mean(tf.square(self.pred_reward - self.inp_r), axis=0)

                self.max_value_constraint = max_value_constraint
                self.value_constraint = value_constraint

                #product_negation_term = 1 if self.reward_mode == 'SUM' else -1
                #product_negation_term = 1
                if self.reward_mode == 'SUM':
                    self.loss = (value_constraint - self.max_value_mult*max_value_constraint)
                else:
                    self.loss = tf.reduce_mean(prod_diff_constraint)

                reward_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'{name}/reward_partition/')
                pred_reward_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'{name}/pred_reward/')
                visual_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.visual_scope.name) if self.visual_scope is not None else []
                print('reward_params', reward_params)
                print('visual_params', visual_params)

                self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, var_list=reward_params + visual_params)
                self.train_op_reward = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.reward_loss, var_list=pred_reward_params)

            all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=f'{name}/')
            print('reward_vars', all_variables)
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

    def train_predicted_reward(self):
        batch_size = 32
        _, _, r_no_reward_batch, sp_no_reward_batch, _ = self.buffer.sample(batch_size // 2)
        _, _, r_reward_batch, sp_reward_batch, _ = self.reward_buffer.sample(batch_size // 2)
        r_batch = r_no_reward_batch + r_reward_batch
        sp_batch = sp_no_reward_batch + sp_reward_batch
        threshold = self.threshold if not self.use_ideal_threshold else self.ideal_threshold
        sp_batch = np.reshape(threshold, [1, 64, 64, 1]) * np.array(sp_batch)
        [_, loss] = self.sess.run([self.train_op_reward, self.reward_loss], feed_dict={self.inp_r: r_batch, self.inp_sp: sp_batch})
        return loss


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

        dummy_env_cluster('reset', args=[])
        starting_states = [[x] for x in dummy_env_cluster('get_current_state', args=[])]
        SP_mixed, R_mixed, T_mixed = self.get_trajectory(dummy_env_cluster, starting_states, -1, self.traj_len)
        feed_dict[self.inp_sp_mixed_traj] = SP_mixed
        feed_dict[self.inp_r_mixed_traj] = R_mixed
        feed_dict[self.inp_t_mixed_traj] = T_mixed

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

    def get_representations(self, s):
        [c0, c1, c2] = self.sess.run([self.all_layers['c0'], self.all_layers['c1'], self.all_layers['c2']], feed_dict={self.inp_sp: [s]})
        return c0[0], c1[0], c2[0]


    # grab sample trajectories from a starting state.
    def get_trajectory(self, dummy_env_cluster, starting_states, policy, trajectory_length):
        if policy == -1:
            policy_func = lambda s_list, i: self.Q_networks[np.random.randint(0, self.num_partitions)].get_action(s_list)
        else:
            policy_func = lambda s_list, i: self.Q_networks[policy].get_action(s_list)
        sp_traj = []
        t_traj = []
        r_traj = []
        #s0 = dummy_env.restore_state(starting_state)
        s_list = dummy_env_cluster('restore_state', sharded_args=starting_states)
        for i in range(trajectory_length):
            #a = self.Q_networks[policy].get_action([s0])[0]
            a_list = [[x] for x in policy_func(s_list, i)]
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


    def reward_visual_tf(self, sp, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            x = sp
            c0 = tf.layers.conv2d(x, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c0')  # [bs, 32, 32, 32]
            c1 = tf.layers.conv2d(c0, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c1')  # [bs, 16, 16, 32]
            c2 = tf.layers.conv2d(c1, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c2')  # [bs, 8, 8, 32]
            internal_rep = tf.reshape(c2, [-1, 8*8*32])
            fc1 = tf.layers.dense(internal_rep, 128, activation=tf.nn.relu, name='fc1')
            r = tf.reshape(tf.layers.dense(fc1, 1, name='pred_reward'), [-1])
        return r, internal_rep, {'c0': c0, 'c1': c1, 'c2': c2}

    # def reward_visual_tf(self, sp, name, reuse=None):
    #     with tf.variable_scope(name, reuse=reuse):
    #         x = sp
    #         num_objects = 4
    #         X, Y = np.meshgrid(np.linspace(0, 1, num=64), np.linspace(0, 1, num=64))
    #         X = X.astype(np.float32)
    #         Y = Y.astype(np.float32)
    #         X = tf.reshape(X, [1, 64, 64])
    #         Y = tf.reshape(Y, [1, 64, 64])
    #         detectors = tf.layers.conv2d(x, 32, 4, 1, activation=tf.nn.relu, padding='SAME', name='d1')
    #         detectors = tf.layers.conv2d(detectors, 2*num_objects, 4, 1, padding='SAME', name='d2')
    #         coords = []
    #         for i in range(num_objects):
    #             object_slice = detectors[:, :, :, 2*i:2*(i+1)]
    #             print(object_slice)
    #             object_slice_X = tf.reshape(tf.nn.softmax(tf.reshape(object_slice[:, :, :, 0], [-1, 64*64])), [-1, 64, 64])
    #             object_slice_Y = tf.reshape(tf.nn.softmax(tf.reshape(object_slice[:, :, :, 1], [-1, 64*64])), [-1, 64, 64])
    #             EX = tf.reshape(tf.reduce_sum(object_slice_X * X, axis=[1,2]), [-1, 1])
    #             EY = tf.reshape(tf.reduce_sum(object_slice_Y * Y, axis=[1,2]), [-1, 1])
    #             coords.append(EX)
    #             coords.append(EY)
    #         coords = tf.concat(coords, axis=1)
    #         fc1 = tf.layers.dense(coords, 128, activation=tf.nn.relu, name='fc1')
    #         r = tf.reshape(tf.layers.dense(fc1, 1, name='pred_reward'), [-1])
    #         return r, coords, None





    def partitioned_reward_tf_visual(self, sp, r, name, reuse=None):
        if self.reuse_visual_scoping:
            x, _ = self.Q_networks[0].qa_network_preprocessing(sp, self.visual_scope, reuse=True)
            # TODO expected behavior should be that the reward network has no control over the visual representation.
            # This should prevent the reward network from fixating on details that are unimportant to the values in an
            # effort to disentangle.
            with tf.variable_scope(name, reuse=reuse):
                soft = tf.layers.dense(x, len(self.Q_networks), activation=tf.nn.softmax, name='qa')
                rewards = tf.reshape(r, [-1, 1]) * soft #* error_control

        elif self.separate_reward_repr:
            _, x, _ = self.reward_visual_tf(sp, 'pred_reward', reuse=True)
            with tf.variable_scope(name, reuse=reuse):
                soft = tf.layers.dense(x, len(self.Q_networks), activation=tf.nn.softmax, name='qa')
                rewards = tf.reshape(r, [-1, 1]) * soft #* error_control

        else:
            with tf.variable_scope(name, reuse=reuse):
                # sp : [bs, 32, 32 ,3]
                print('r', r)
                x = sp
                x = tf.layers.conv2d(x, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c0') # [bs, 32, 32, 32]
                x = tf.layers.conv2d(x, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c1')  # [bs, 16, 16, 32]
                x = tf.layers.conv2d(x, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c2')  # [bs, 8, 8, 32]
                x = tf.layers.dense(tf.reshape(x, [-1, 8 * 8 * 32]), 128, activation=tf.nn.relu, name='fc1')
                if self.reward_mode == 'SUM':
                    soft = tf.layers.dense(x, len(self.Q_networks), activation=tf.nn.softmax, name='qa')
                    rewards = tf.reshape(r, [-1, 1]) * soft #* error_control
                else:
                    soft = tf.layers.dense(x, len(self.Q_networks), activation=tf.nn.softmax, name='qa')
                    rewards = tf.exp(tf.reshape(tf.log(r + 0.01), [-1, 1]) * soft)  # * error_control
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

    def update_threshold_image(self, threshold):
        self.threshold = np.copy(threshold)

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

