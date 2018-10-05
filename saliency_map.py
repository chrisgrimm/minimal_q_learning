import tensorflow as tf
import numpy as np
import cv2
from typing import List, Callable
import itertools
from utils import build_directory_structure
import os


class LatentAttentionNetwork(object):

    def __init__(self,
                 build_network_input: Callable[[tf.Tensor], tf.Tensor],
                 image_shape: List[int],
                 name: str,
                 num_partitions: int,
                 reuse=None,
                 sess=None):
        image_shape = list(image_shape)
        self.num_partitions = num_partitions
        self.inp_image = inp_image = tf.placeholder(tf.uint8, shape=image_shape, name='inp_image')
        self.inp_partition = tf.placeholder(tf.uint8)
        reward_onehot = tf.one_hot(self.inp_partition, num_partitions) # [num_partitions]

        inp_image_converted = tf.image.convert_image_dtype(inp_image, dtype=tf.float32)
        inp_noise = tf.random_uniform(shape=image_shape, minval=0, maxval=1, dtype=tf.float32)
        beta = 1.0

        with tf.variable_scope(name, reuse=reuse):
            self.scope = tf.get_variable_scope()
            # make a 2D mask for the pixels.
            mask = mask_var = tf.get_variable('mask', shape=image_shape[:2], dtype=tf.float32,
                                              initializer=tf.constant_initializer())
            self.mask = mask
            mask = tf.nn.sigmoid(tf.reshape(mask, image_shape[:2] + [1]))
            # when the mask is 1 this is noise.
            corrupted_image = inp_image_converted * (1 - mask) + inp_noise * mask
            #corrupted_image = inp_noise
        # build the network components. this functions need to manage their scopes correctly.
        output = build_network_input(inp_image_converted) # [1, num_partitions]
        # TODO: we need to make the inputs floats because of the corruption.
        corrupted_output = build_network_input(corrupted_image) # [1, num_partitions]

        with tf.variable_scope(name, reuse=reuse):
            sq_diff = tf.Print(tf.square(output - corrupted_output), [tf.square(output - corrupted_output)], message='sq_diff')
            self.loss = loss = tf.reduce_mean(sq_diff) - beta * tf.reduce_mean(tf.square(mask))
            self.train_op = train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name))
        self.all_vars = all_vars = tf.get_collection(tf.GraphKeys.VARIABLES, self.scope.name)
        if sess is None:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = sess = tf.Session(config=config)
        else:
            self.sess = sess
        sess.run(tf.initialize_variables(all_vars))


    def train_mask(self,
                   image: np.ndarray,
                   path: str,
                   partition_num: int,
                   save_freq: int):
        assert 0 <= partition_num < self.num_partitions
        self.sess.run(tf.initialize_variables(self.all_vars))
        build_directory_structure(path, {'masks': {f'mask_{partition_num}': {}}})
        cv2.imwrite(os.path.join(path, 'image.png'), image[:, :, 6:9])
        for time in itertools.count(0):
            [_, loss, mask] = self.sess.run([self.train_op, self.loss, self.mask], feed_dict={self.inp_image: image,
                                                                                              self.inp_partition: partition_num})
            print(f'({time}) loss: {loss}')
            if time % save_freq == 0:
                file = os.path.join(path, 'masks', f'mask_{partition_num}', f'mask_{time}.png')
                print(file)
                mask = 255*np.tile(np.reshape(mask, list(mask.shape[:2])+[1]), [1,1,3])
                cv2.imwrite(file, mask)


if __name__ == '__main__':
    from reward_network import RewardPartitionNetwork
    from envs.atari.pacman import PacmanWrapper
    path = 'pacman_new_dqn_5x_freq/part2'
    name = 'reward_net.ckpt'
    env = PacmanWrapper()
    s = env.reset()
    r = 0
    while r != 1:
        s, r, t, _ = env.step(np.random.randint(0, env.action_space.n))
    num_partitions = 2
    num_visual_channels = 9
    reward_net = reward_net = RewardPartitionNetwork(env, None, None, num_partitions, env.observation_space.shape[0],
                                        env.action_space.n, 'reward_net', traj_len=10,
                                        num_visual_channels=num_visual_channels, visual=True, gpu_num=-1)
    reward_net.restore(path, name)

    def network_hook(state):
        reward = reward_net.partitioned_reward_tf(tf.reshape(state, [1, 64, 64, 9]), tf.ones([1], dtype=tf.float32), name='reward_net/reward_partition', reuse=True)
        print('reward', reward)
        return tf.reshape(reward, [num_partitions])

    lan = LatentAttentionNetwork(network_hook, env.observation_space.shape, 'lan', num_partitions, sess=reward_net.sess)
    lan.train_mask(s, 'pacman_new_dqn_5x_freq/', 0, save_freq=1000)









