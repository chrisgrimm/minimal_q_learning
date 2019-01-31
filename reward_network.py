import tensorflow as tf
import numpy as np
import os
from baselines.deepq.experiments.training_wrapper import make_dqn

EPS = 10**-6

class RewardPartitionNetwork(object):

    def __init__(self, env, buffer, state_buffer, num_partitions, obs_size, num_actions, name, traj_len=30,
                 max_value_mult=10, use_dynamic_weighting_max_value=True, use_dynamic_weighting_disentangle_value=False,
                 visual=False, num_visual_channels=3, gpu_num=0, use_gpu=False, lr=0.0001, reuse=None,
                 softmin_temperature=1.0, product_mode=False,
                 stop_softmin_gradients=True):
        if not use_gpu:
            gpu_num = 0

        assert traj_len % 2 == 0
        self.product_mode = product_mode
        self.state_buffer = state_buffer
        self.softmin_temperature = softmin_temperature
        self.stop_softmin_gradients = stop_softmin_gradients

        self.num_partitions = num_partitions
        self.num_actions = num_actions
        self.obs_size = obs_size
        self.buffer = buffer
        self.visual = visual
        self.traj_len = traj_len
        self.max_value_mult = max_value_mult
        self.use_dynamic_weighting_max_value = use_dynamic_weighting_max_value
        self.use_dynamic_weighting_disentangle_value = use_dynamic_weighting_disentangle_value

        self.num_visual_channels = num_visual_channels
        self.obs_shape = [None, self.obs_size] if not self.visual else [None, 64, 64, self.num_visual_channels]
        self.obs_shape_traj = [None, self.traj_len, self.obs_size] if not self.visual else [None, self.traj_len, 64, 64, self.num_visual_channels]
        self.visual_scope = None
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)


        self.Q_networks = [make_dqn(env, f'qnet{i}', gpu_num) for i in range(num_partitions)]

        with tf.device(f'/{"gpu" if use_gpu else "cpu"}:{gpu_num}'):
            with tf.variable_scope(name, reuse=reuse):
                #self.inp_only_rewarding_trajectories = tf.placeholder(tf.bool)
                self.inp_sp = tf.placeholder(tf.uint8, self.obs_shape)
                inp_sp_converted = tf.image.convert_image_dtype(self.inp_sp, dtype=tf.float32)
                self.inp_r = tf.placeholder(tf.float32, [None])

                partitioned_reward = self.partitioned_reward_tf(inp_sp_converted, self.inp_r, 'reward_partition') # [bs, num_partitions]
                self.partitioned_reward = partitioned_reward
                self.combined_reward = tf.reduce_sum(tf.cumprod(self.partitioned_reward, axis=1), axis=1)

                self.inp_sp_traj = tf.placeholder(tf.uint8, self.obs_shape_traj)
                converted_inp_sp_traj = tf.image.convert_image_dtype(self.inp_sp_traj, dtype=tf.float32)
                self.inp_r_traj = tf.placeholder(tf.float32, [None, self.traj_len])
                self.inp_t_traj = tf.placeholder(tf.bool, [None, self.traj_len])
                partitioned_traj = self.partition_reward_traj(converted_inp_sp_traj, self.inp_r_traj, 'reward_partition', reuse=True)
                # partitioned_traj : [bs, traj_len, num_partitions]
                assert self.num_partitions == 2
                P = partitioned_traj
                combined_traj_1 = P[:,:,0] + P[:,:,0]*P[:,:,1]
                combined_traj_2 = P[:,:,1] + P[:,:,1]*P[:,:,0]
                #combined_traj = tf.reduce_sum(tf.cumprod(partitioned_traj, axis=2), axis=2) # [bs, traj_len]
                #combined_traj_2 =

                value1 = self.get_values(tf.reshape(combined_traj_1, [-1, self.traj_len, 1]), self.inp_t_traj, self.traj_len)
                value1 = tf.reshape(value1, [-1])
                value2 = self.get_values(tf.reshape(combined_traj_2, [-1, self.traj_len, 1]), self.inp_t_traj,
                                         self.traj_len)
                value2 = tf.reshape(value1, [-1])
                self.loss = tf.reduce_mean(tf.maximum(value1, value2))

                reward_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=f'{name}/reward_partition/')
                visual_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.visual_scope.name) if self.visual_scope is not None else []
                opt = tf.train.AdamOptimizer(learning_rate=lr)
                gradients = opt.compute_gradients(self.loss, var_list=reward_params + visual_params)

                self.train_op = opt.apply_gradients(gradients)
                #self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, var_list=reward_params + visual_params)

            all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=f'{name}/')
            self.saver = tf.train.Saver(var_list=all_variables)
            #print('reward_vars', all_variables)
            self.sess.run(tf.variables_initializer(all_variables))

    def save(self, path, name):
        self.saver.save(self.sess, os.path.join(path, name))
        for i, q_net in enumerate(self.Q_networks):
            q_net.save(path, f'q_net{i}.ckpt')

    def restore(self, path, name):
        self.saver.restore(self.sess, os.path.join(path, name))
        for i, q_net in enumerate(self.Q_networks):
            q_net.restore(path, f'q_net{i}.ckpt')

    #def train_Q_network(self):


    # def train_Q_networks(self, time):
    #     Q_losses = []
    #     s_batch, a_batch, r_batch, sp_batch, t_batch = self.buffer.sample(32)
    #     [partitioned_reward] = self.sess.run([self.partitioned_reward], feed_dict={self.inp_sp: sp_batch, self.inp_r: r_batch})
    #     for i, network in enumerate(self.Q_networks):
    #         weights, batch_idxes = np.ones_like(t_batch), None
    #         loss = network.train_batch(time, s_batch, a_batch, partitioned_reward[:, i], sp_batch, t_batch, weights, batch_idxes)
    #         Q_losses.append(loss)
    #     return Q_losses




    def train_R_function(self, inp_s_traj, inp_r_traj, inp_t_traj):
        # inp_s_traj : [big_traj_len, 64, 64, 3]
        # inp_r_traj : [big_traj_len]
        big_traj_len = inp_r_traj.shape[0]

        batch_size = 32
        starting_points = np.random.randint(0, big_traj_len - self.traj_len, size=[batch_size])
        ending_points = starting_points + self.traj_len
        batch_sp = []
        batch_r = []
        batch_t = []
        for start, end in zip(starting_points, ending_points):
            batch_sp.append(inp_s_traj[start:end, :, :, :])
            batch_r.append(inp_r_traj[start:end])
            batch_t.append(inp_t_traj[start:end])

        feed_dict = {
            self.inp_sp_traj: batch_sp,
            self.inp_r_traj: batch_r,
            self.inp_t_traj: batch_t,
        }
        [_, loss] = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss



    # grab sample trajectories from a starting state.
    def get_trajectory(self, dummy_env_cluster, starting_states, policy, trajectory_length):
        sp_traj = []
        t_traj = []
        r_traj = []
        def get_action(policy, s_list):
            if policy >= 0:
                return self.Q_networks[policy].get_action(s_list)
            else:
                return np.random.randint(0, self.num_actions, size=[len(s_list)])
        #s0 = dummy_env.restore_state(starting_state)
        init_s_list = s_list = dummy_env_cluster('restore_state', sharded_args=starting_states)
        for i in range(trajectory_length):
            #a = self.Q_networks[policy].get_action([s0])[0]
            a_list = [[x] for x in get_action(policy, s_list)]
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
        return sp_traj, r_traj, t_traj, init_s_list


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
            #print('r', r)
            x = sp
            x = tf.layers.conv2d(x, 32, 8, 4, 'SAME', activation=tf.nn.relu, name='c0') # [bs, 16, 16, 32]
            x = tf.layers.conv2d(x, 64, 4, 2, 'SAME', activation=tf.nn.relu, name='c1')  # [bs, 8, 8, 64]
            x = tf.layers.conv2d(x, 64, 3, 1, 'SAME', activation=tf.nn.relu, name='c2')  # [bs, 8, 8, 64]
            x = tf.layers.dense(tf.reshape(x, [-1, 8 * 8 * 64]), 128, activation=tf.nn.relu, name='fc1')
            soft = tf.layers.dense(x, len(self.Q_networks), activation=tf.nn.softmax, name='qa')
        # check this...
        if self.product_mode:
            rewards = tf.exp(tf.reshape(tf.log(r + EPS), [-1, 1]) * soft) #* error_control
        else:
            rewards = tf.reshape(r, [-1, 1]) * soft #* error_control
        return rewards

    def partition_reward_traj(self, sp_traj, r_traj, name, reuse=None):
        Rs_traj = []
        for t in range(self.traj_len):
            #s = s_traj[:, t, :]
            #a = a_traj[:, t, :]
            sp = sp_traj[:, t, :]
            r = r_traj[:, t]
            #print('r', r)
            Rs = self.partitioned_reward_tf(sp, r, name, reuse=(t > 0) or (reuse == True))
            Rs_traj.append(tf.reshape(Rs, [-1, 1, self.num_partitions])) # [bs, 1, n]
        return tf.concat(Rs_traj, axis=1) # [bs, traj, n]

    def get_partitioned_reward(self, sp, r):
        [partitioned_r]  = self.sess.run([self.partitioned_reward], feed_dict={self.inp_sp: sp, self.inp_r: r})
        return partitioned_r

    def get_combined_reward(self, sp, r):
        [combined_reward] = self.sess.run([self.combined_reward], feed_dict={self.inp_sp: sp, self.inp_r: r})
        return combined_reward

    def get_state_values(self, s):
        return [np.max(self.Q_networks[i].get_Q(s), axis=1) for i in range(self.num_partitions)]

    def get_state_actions(self, s):
        return [self.Q_networks[i].get_action(s) for i in range(self.num_partitions)]

    def get_reward(self, sp, r):
        return self.get_partitioned_reward([sp], [r])[0]

    # returns vector V where V_i = value of trajectory under reward i.
    def get_values(self, rs_traj, ts_traj, traj_len):
        # rs_traj : [bs, traj_len, num_partitions]
        # ts_traj : [bs, traj_len]
        #print(rs_traj)
        gamma = 0.99
        gamma_sequence = tf.reshape(tf.pow(gamma, list(range(traj_len))), [1, traj_len, 1])
        t_sequence = 1.0 - tf.reshape(tf.cast(ts_traj, tf.float32), [-1, traj_len, 1])
        # after the first terminal state, sticky_t_sequence should always be 0.
        sticky_t_sequence = tf.cumprod(t_sequence, axis=1)
        #prod_reward = 0.0
        out = tf.reduce_sum(rs_traj * gamma_sequence * sticky_t_sequence, axis=1) # [bs, num_partitions]
        #print('out', out)
        return out

