import tensorflow as tf
import numpy as np
from constants import *

class Generator:

    loss = None
    trainer = None
    generated_image = None

    def __init__(self,batch_size, z_dim):
        self.generated_image = self._create_graph(batch_size,z_dim)

    def get_generated_image_tensor(self):
        return self.generated_image

    def _track_variables(self):
        tf.summary.scalar('Generator_loss', self.loss)

    def _create_graph(self, batch_size, z_dim):
        z = tf.random_normal([batch_size, z_dim], mean=0, stddev=1, name='z')
        g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g1 = tf.matmul(z, g_w1) + g_b1
        g1 = tf.reshape(g1, [-1, 56, 56, 1])
        g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
        g1 = tf.nn.relu(g1)

        # Generate 50 features
        g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
        g2 = g2 + g_b2
        g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
        g2 = tf.nn.relu(g2)
        g2 = tf.image.resize_images(g2, [56, 56])

        # Generate 25 features
        g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
        g3 = g3 + g_b3
        g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
        g3 = tf.nn.relu(g3)
        g3 = tf.image.resize_images(g3, [56, 56])

        # Final convolution with one output channel
        g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
        g4 = g4 + g_b4
        g4 = tf.sigmoid(g4)

        # No batch normalization at the final layer, but we do add
        # a sigmoid activator to make the generated images crisper.
        # Dimensions of g4: batch_size x 28 x 28 x 1

        return g4

    def _create_loss_functions(self,output_logits):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_logits, labels=tf.ones_like(output_logits)))

    def get_trainer(self):
        return self.trainer

    def get_loss(self):
        return self.loss

    def setup_optimizer(self,output_logits):
        self._create_loss_functions(output_logits)
        all_vars = tf.trainable_variables()
        my_vars = [var for var in all_vars if 'g_' in var.name]
        with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
	           self.trainer = tf.train.AdamOptimizer(0.004).minimize(self.loss, var_list=my_vars)
        self._track_variables()
