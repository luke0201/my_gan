import os
import time
import numpy as np
import tensorflow as tf

import model
from data_loader import MNISTLoader

EPS = 1e-3

latent_size = 100
n_epoch = 100
batch_size = 200
log_dir = 'log'
checkpoint_dir = 'checkpoint'

Z = tf.placeholder(tf.float32, [batch_size, latent_size], name='Z')
X = tf.placeholder(tf.float32, [batch_size, 784], name='X')

# Preprocess
real_output = tf.reshape(X, [batch_size, 28, 28, 1])
real_image_summary = tf.summary.image('Real Image', real_output, max_outputs=4)
X_preproc = tf.scalar_mul(2. / 255., X)
X_preproc = tf.add(X_preproc, tf.constant(-1., shape=X_preproc.shape))

g_z = model.generator(Z, training=True)
fake_output = tf.reshape(g_z, [batch_size, 28, 28, 1])
tf.summary.image('Generated Image', fake_output, max_outputs=4)
d_real = model.discriminator(X_preproc, fake=False, training=True)
d_fake = model.discriminator(g_z, fake=True, training=True)

# Loss
d_loss = -tf.reduce_mean(tf.log(d_real + EPS) + tf.log(1. - d_fake + EPS))
tf.summary.scalar('Discriminator Loss', d_loss)
g_loss = -tf.reduce_mean(tf.log(d_fake))
tf.summary.scalar('Generator Loss', g_loss)

# Train ops
optimizer_d = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
optimizer_g = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    var_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    train_d = optimizer_d.minimize(d_loss, var_list=var_d)
    var_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    train_g = optimizer_g.minimize(g_loss, var_list=var_g)

def main():
    mnist = MNISTLoader()
    print('Dataset loaded\n')

    saver = tf.train.Saver()

    print('Start training\n')
    start_time = time.time()

    with tf.Session() as sess:
        print()

        tf.global_variables_initializer().run()

        with tf.summary.FileWriter(
                os.path.join(log_dir, str(start_time)),
                graph=sess.graph) as writer:

            iter_per_epoch = mnist.size // batch_size
            for iteration in range(1, n_epoch * iter_per_epoch + 1):
                Z_batch = np.random.normal(size=[batch_size, latent_size])
                X_batch, _ = mnist.next_batch(batch_size)

                d_loss_val, _ = sess.run([d_loss, train_d], feed_dict={Z: Z_batch, X: X_batch})
                g_loss_val, _ = sess.run([g_loss, train_g], feed_dict={Z: Z_batch, X: X_batch})

                if iteration % iter_per_epoch == 0:
                    cur_epoch = iteration // iter_per_epoch

                    # Print summary
                    print('Epoch:', cur_epoch)
                    print('- Generator Loss:', g_loss_val)
                    print('- Discriminator Loss:', d_loss_val)

                    summary = tf.summary.merge_all().eval(feed_dict={Z: Z_batch, X: X_batch})
                    writer.add_summary(summary, global_step=cur_epoch)
                    print()

        saver.save(sess, os.path.join(checkpoint_dir, 'final-' + str(start_time)))

    print('Training done')
    end_time = time.time()
    print('Elapsed time: ', end_time - start_time)

if __name__ == '__main__':
    main()
