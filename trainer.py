import tensorflow as tf
import numpy as np
import datetime
from discriminator import *
from generator import *
from constants import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

sess = tf.Session()
real_image_placeholder = tf.placeholder("float", shape = [None,28,28,1], name='image_placeholder')
generator = Generator(batch_size, z_dimensions)
discriminator = Discriminator(real_image_placeholder,generator.get_generated_image_tensor())
generator.setup_optimizer(discriminator.get_generated_image_probability())

d_real_count_ph = tf.placeholder(tf.float32)
d_fake_count_ph = tf.placeholder(tf.float32)
g_count_ph = tf.placeholder(tf.float32)

tf.summary.scalar('d_real_count', d_real_count_ph)
tf.summary.scalar('d_fake_count', d_fake_count_ph)
tf.summary.scalar('g_count', g_count_ph)

images_for_tensorboard = generator.get_generated_image_tensor()
tf.summary.image('Generated_images', images_for_tensorboard, 10)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
sess.run(tf.global_variables_initializer())

# == TRAINING LOOP ==
gLoss = 1
dLossFake, dLossReal = 0, 0
d_real_count, d_fake_count, g_count = 0, 0, 0

for i in range(50000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    if dLossFake > 0.7:
        # Train discriminator on generated images
        _, dLossReal, dLossFake, gLoss = sess.run([discriminator.get_fake_trainer(), discriminator.get_real_loss(), discriminator.get_fake_loss(), generator.get_loss()],
                                                    {real_image_placeholder: real_image_batch})
        d_fake_count += 1

    if gLoss > 0.5:
        # Train the generator
        _, dLossReal, dLossFake, gLoss = sess.run([generator.get_trainer(), discriminator.get_real_loss(),  discriminator.get_fake_loss(),  generator.get_loss()],
                                                    {real_image_placeholder: real_image_batch})
        g_count += 1

    if dLossReal > 0.45:
        # If the discriminator classifies real images as fake,
        # train discriminator on real values
        _, dLossReal, dLossFake, gLoss = sess.run([discriminator.get_real_trainer(), discriminator.get_real_loss(), discriminator.get_fake_loss(), generator.get_loss()],
                                                    {real_image_placeholder: real_image_batch})
        d_real_count += 1

    if i % 10 == 0:
        real_image_batch = mnist.validation.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])

        summary = sess.run(merged, {real_image_placeholder: real_image_batch, d_real_count_ph: d_real_count,
                                    d_fake_count_ph: d_fake_count, g_count_ph: g_count})
        writer.add_summary(summary, i)
        d_real_count, d_fake_count, g_count = 0, 0, 0

    if i % 5000 == 0:
        save_path = saver.save(sess, "models/pretrained_gan.ckpt", global_step=i)
        print("saved to %s" % save_path)
