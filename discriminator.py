import tensorflow as tf
import numpy as np
from constants import *

class Discriminator:

    input_image = None
    loss_real = None
    loss_fake = None
    loss = None
    trainer_fake = None
    trainer_real = None
    real_image_placeholder = None
    generated_image_placeholder = None
    generated_image_graph = None
    real_image_graph = None

    def __init__(self, real_image_placeholder, generated_image_placeholder):
        self.real_image_placeholder = real_image_placeholder
        self.generated_image_placeholder = generated_image_placeholder
        self._setup_network()
        self._track_variables()

    def _track_variables(self):
        tf.summary.scalar('Discriminator_loss_real', self.loss_real)
        tf.summary.scalar('Discriminator_loss_fake', self.loss_fake)

    def _setup_network(self):
        self.real_image_graph = self._create_graph(self.real_image_placeholder,False)
        self.generated_image_graph = self._create_graph(self.generated_image_placeholder,True)
        self.setup_optimizer()

    def get_real_loss(self):
        return self.loss_real

    def get_fake_loss(self):
        return self.loss_fake

    def get_fake_trainer(self):
        return self.trainer_fake

    def get_real_trainer(self):
        return self.trainer_real

    def get_generated_image_probability(self):
        return self.generated_image_graph

    def get_real_image_probability(self):
        return self.real_image_graph

    def _create_graph(self,input_image,reuse_vars):
        if (reuse_vars):
            tf.get_variable_scope().reuse_variables()

        d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=input_image, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))

        d4 = tf.matmul(d3, d_w4) + d_b4
        return d4

    def create_loss_function(self):
        self.loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_image_graph, labels=tf.fill([batch_size, 1], 0.9)))
        self.loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.generated_image_graph, labels=tf.zeros_like(self.generated_image_graph)))
        self.loss = self.loss_real + self.loss_fake

    def setup_optimizer(self):
        self.create_loss_function()
        all_vars = tf.trainable_variables()
        my_vars = [var for var in all_vars if 'd_' in var.name]
        with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
            self.trainer_fake = tf.train.AdamOptimizer(0.001).minimize(self.loss_fake, var_list=my_vars)
            self.trainer_real = tf.train.AdamOptimizer(0.001).minimize(self.loss_real, var_list=my_vars)
