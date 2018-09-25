import tensorflow as tf
import numpy as np
import os


class QLearnerAgent(object):
    # classes have name scopes. need a way to allow an agent to use another agent's weights for a portion of the network.
    # this needs to be suppliable as a function, we need a function that will produce a tensorflow graph
    def __init__(self, obs_size, num_actions, name, reuse=None, visual=False, num_visual_channels=3, gpu_num=0, use_gpu=False,
                 alternate_visual_scope=None, sess=None):
        if not use_gpu:
            gpu_num = 0
        with tf.device(f'/{"gpu" if use_gpu else "cpu"}:{gpu_num}'):
            with tf.variable_scope(name, reuse=reuse):
                print(obs_size, num_actions)
                self.obs_size = obs_size
                self.num_actions = num_actions
                self.gamma = 0.99
                self.use_top_level=False
                self.visual = visual
                tau = 0.998
                self.state_shape = [None, obs_size] if not visual else [None, 64, 64, num_visual_channels]

                if self.visual:
                    self.inp_s = tf.placeholder(tf.uint8, self.state_shape, name='inp_s')
                    self.inp_s_converted = tf.image.convert_image_dtype(self.inp_s, dtype=tf.float32)
                    self.inp_sp = tf.placeholder(tf.uint8, self.state_shape, name='inp_sp')
                    self.inp_sp_converted = tf.image.convert_image_dtype(self.inp_sp, dtype=tf.float32)
                else:
                    self.inp_s = tf.placeholder(tf.float32, self.state_shape, name='inp_s')
                    self.inp_s_converted = self.inp_s
                    self.inp_sp = tf.placeholder(tf.float32, self.state_shape, name='inp_sp')
                    self.inp_sp_converted = self.inp_sp

                self.inp_a = tf.placeholder(tf.int32, [None], name='inp_a')
                inp_a_onehot = tf.one_hot(self.inp_a, self.num_actions)
                self.inp_r = tf.placeholder(tf.float32, [None], name='inp_r')
                self.inp_t = tf.placeholder(tf.float32, [None], name='inp_t')
                if alternate_visual_scope is not None:
                    # TODO confirm that this works with reuse.
                    # a quick look at the source-code suggests that it should.
                    print('alt_scope', alternate_visual_scope)
                    self.q_encoding, self.visual_scope = self.qa_network_preprocessing(self.inp_s_converted, alternate_visual_scope, reuse=True)
                else:
                    print('here!')
                    self.q_encoding, self.visual_scope = self.qa_network_preprocessing(self.inp_s_converted, 'qa_network_preprocessing')
                self.qa = qa = self.qa_network(self.q_encoding, 'qa_network_out')
                # TODO confirm that this works as expected. and that reused variables are being updated properly.
                preprocessing_scope = f'{name}/qa_network_preprocessing/' if alternate_visual_scope is None else alternate_visual_scope.name
                qa_vars = (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope=preprocessing_scope) +
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope=f'{name}/qa_network_out/'))

                print('qa_vars', qa_vars)

                qa_target_encoding, _ = self.qa_network_preprocessing(self.inp_sp_converted, 'qa_target_preprocessing')
                qa_target = self.qa_network(qa_target_encoding, 'qa_target_out')
                qa_target_vars = (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope=f'{name}/qa_target_preprocessing/') +
                                  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope=f'{name}/qa_target_out/'))

                y = self.inp_r + (1 - self.inp_t) * self.gamma * tf.reduce_max(qa_target, axis=1)
                self.loss = tf.reduce_mean(tf.square(tf.stop_gradient(y) - tf.reduce_sum(qa * inp_a_onehot, axis=1)))
                optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)
                gvs = optimizer.compute_gradients(self.loss, var_list=qa_vars)
                for grad, var in gvs:
                    print(grad, var)
                capped_gvs = [(None if grad is None else tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                #capped_gvs = gvs
                self.train_op = optimizer.apply_gradients(capped_gvs)

                # variable update operations
                soft_update_target = tf.group(*[tf.assign(target, tau*target + (1 - tau)*network)
                                                for network, target in zip(qa_vars, qa_target_vars)])
                self.soft_update_target = soft_update_target
                hard_update_target = tf.group(*[tf.assign(target, network)
                                                for network, target in zip(qa_vars, qa_target_vars)])
                self.hard_update_target = hard_update_target
                if sess is None:
                    config = tf.ConfigProto(allow_soft_placement=True)
                    config.gpu_options.allow_growth = True
                    self.sess = tf.Session(config=config)
                else:
                    self.sess = sess
            all_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope=name+'/')
            print('all_vars', all_vars)
            self.saver = tf.train.Saver(var_list=all_vars)
            self.sess.run(tf.variables_initializer(all_vars))
            self.sess.run(self.hard_update_target)
            print(f'Finished Initializing {name}')

    def save(self, path, name):
        self.saver.save(self.sess, os.path.join(path, name))

    def restore(self, path, name):
        self.saver.restore(self.sess, os.path.join(path, name))


    def qa_network_preprocessing(self, obs, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            scope = tf.get_variable_scope()
            if self.visual:
                x = obs
                x = tf.layers.conv2d(x, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c0')  # [bs, 32, 32, 32]
                x = tf.layers.conv2d(x, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c1')  # [bs, 16, 16, 32]
                x = tf.layers.conv2d(x, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c2')  # [bs, 8, 8, 32]
                x = tf.layers.dense(tf.reshape(x, [-1, 8 * 8 * 32]), 256, activation=tf.nn.relu, name='fc1')
                return x, scope
            else:
                fc1 = tf.layers.dense(obs, 128, tf.nn.relu, name='fc1')
                fc2 = tf.layers.dense(fc1, 128, tf.nn.relu, name='fc2')
                return fc2, scope

    def qa_network(self, preprocessed_obs, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            qa = tf.layers.dense(preprocessed_obs, self.num_actions, name='qa')
        return qa


    def encode_q(self, obs):
        [encoding] = self.sess.run([self.q_encoding], feed_dict={self.inp_s: obs})
        return encoding

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

    def get_Q(self, s):
        [qa] = self.sess.run([self.qa], feed_dict={self.inp_s: s})
        return qa




