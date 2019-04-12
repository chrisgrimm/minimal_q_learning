import tensorflow as tf
import numpy as np
from utils import horz_stack_images
import cv2
from itertools import count
EPS = 10**-6

def make_n_columns(height_percents, spacing=2, size=32):
    background = np.zeros((size, size, 3), dtype=np.float32)
    num_columns = len(height_percents)
    column_width = (size - (num_columns-1)*spacing) // num_columns
    assert column_width > 0
    start_pos = 0
    for i in range(num_columns):
        height = int(height_percents[i]*size)
        background[size-height:size, start_pos:start_pos+column_width, :] = 1
        start_pos += (column_width + spacing)
    return background

def get_batch_n_columns(batch_size, size=32, num_columns=3, spacing=2):
    images = []
    for i in range(batch_size):
        images.append(make_n_columns(np.random.uniform(0, 1, size=num_columns), spacing=spacing, size=size))
    return np.array(images)



class GNetwork:

    def __init__(self):
        self.inp_image = tf.placeholder(tf.float32, [None, 32, 32, 3])

        self.inp_merge_left = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.inp_merge_right = tf.placeholder(tf.float32, [None, 32, 32, 3])

        self.encoding_size = 10

        enc, recon, ae_vars = self.build_autoencoder(self.inp_image, 'autoencoder')
        enc_left, _, _ = self.build_autoencoder(self.inp_merge_left, 'autoencoder', reuse=True)
        enc_right, _, _ = self.build_autoencoder(self.inp_merge_right, 'autoencoder', reuse=True)

        merged = tf.concat([enc_left[:, :self.encoding_size//2], enc_right[:, self.encoding_size//2:]], axis=1) # [bs, self.encoding_size]
        self.merged_decoded, _ = self.build_decoder(merged, 'autoencoder', reuse=True)


        X, Y = enc[:, :self.encoding_size//2], enc[:, self.encoding_size//2:]


        mu_Y_given_X, logvar_Y_given_X, q1_vars = self.build_mean_std(tf.stop_gradient(X), 'q1')
        mu_X_given_Y, logvar_X_given_Y, q2_vars = self.build_mean_std(tf.stop_gradient(Y), 'q2')
        q1_pdf = self.log_pdf(Y, mu_Y_given_X, logvar_Y_given_X)
        q2_pdf = self.log_pdf(X, mu_X_given_Y, logvar_X_given_Y)

        self.q1_pdf = q1_pdf
        self.q2_pdf = q2_pdf

        self.ae_loss = tf.reduce_mean(tf.square(recon - self.inp_image))
        self.G_loss = tf.reduce_mean(-q1_pdf - q2_pdf, axis=0)

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.ae_loss + 0.01*self.G_loss, var_list=ae_vars+q1_vars+q2_vars)
        #self.train_q = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(-self.G_loss, var_list=q1_vars+q2_vars)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



    def train(self, batch_size=32):
        X1 = get_batch_n_columns(batch_size, num_columns=2)
        [_, ae_loss, G_loss, q1_pdf, q2_pdf] = self.sess.run([self.train_op, self.ae_loss, self.G_loss, self.q1_pdf, self.q2_pdf], feed_dict={self.inp_image: X1})
        print(np.mean(q1_pdf))
        #[_, G_loss] = self.sess.run([self.train_q, self.G_loss], feed_dict={self.inp_image: X2})
        return ae_loss, G_loss


    def display_results(self):
        column_pairs = [
            ([0.9, 0.0], [0.0, 0.9]),
            ([0.5, 0.9], [0.0, 0.5]),
            ([0.0, 0.9], [0.9, 0.0]),
            ([0.9, 0.5], [0.5, 0.0])
        ]
        X_left = [make_n_columns(heights) for heights, _ in column_pairs]
        X_right = [make_n_columns(heights) for _, heights in column_pairs]
        [all_merged] = self.sess.run([self.merged_decoded], feed_dict={self.inp_merge_left: X_left, self.inp_merge_right: X_right})
        print(all_merged.shape)
        for i, (left, right, merged) in enumerate(zip(X_left, X_right, all_merged)):
            stack = horz_stack_images(left, right, merged, background_color=(255,0,0))
            img = 255*stack
            cv2.imwrite(f'test_images/{i}.png', img)





    def build_autoencoder(self, x, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            c1 = tf.layers.conv2d(x, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c1') # [bs, 16, 16, 32]
            c2 = tf.layers.conv2d(c1, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='c2') # [bs, 8, 8, 32]
            c2_flat = tf.reshape(c2, [-1, 8*8*32])
            encoding = tf.layers.dense(c2_flat, self.encoding_size, activation=tf.nn.tanh, name='encoding')
            d2_flat = tf.layers.dense(encoding, 8*8*32, activation=tf.nn.relu, name='d2_flat')
            d2 = tf.reshape(d2_flat, [-1, 8, 8, 32])
            d1 = tf.layers.conv2d_transpose(d2, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='d1') # [16, 16, 32]
            out = tf.layers.conv2d_transpose(d1, 3, 4, 2, 'SAME',  activation=tf.nn.sigmoid, name='out') # [32, 32, 1]
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.original_name_scope)
        return encoding, out, vars

    def build_decoder(self, encoding, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            d2_flat = tf.layers.dense(encoding, 8 * 8 * 32, activation=tf.nn.relu, name='d2_flat')
            d2 = tf.reshape(d2_flat, [-1, 8, 8, 32])
            d1 = tf.layers.conv2d_transpose(d2, 32, 4, 2, 'SAME', activation=tf.nn.relu, name='d1')  # [16, 16, 32]
            out = tf.layers.conv2d_transpose(d1, 3, 4, 2, 'SAME', activation=tf.nn.sigmoid, name='out')  # [32, 32, 1]
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.original_name_scope)
        return out, vars

    def build_mean_std(self, x, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse) as scope:
            fc1 = tf.layers.dense(x, 128, tf.nn.relu, name='fc1')
            enc = tf.layers.dense(fc1, 128, tf.nn.relu, name='enc')
            enc_mean = tf.layers.dense(enc, self.encoding_size // 2, name='enc_mean')
            enc_std = tf.layers.dense(enc, self.encoding_size // 2, activation=tf.nn.tanh, name='enc_std')
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.original_name_scope)
        return enc_mean, enc_std, vars


    def log_pdf(self, x, mean, logvar):
        return -0.5 * (tf.reduce_sum(logvar, axis=1) + tf.reduce_sum(tf.square(x - mean) / (tf.exp(logvar) + EPS), axis=1)) # [bs]



if __name__ == '__main__':
    net = GNetwork()
    disp_freq = 100
    for i in count():
        print(i, net.train(32))
        if i % disp_freq == 0:
            net.display_results()

