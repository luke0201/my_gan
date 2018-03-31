import os
from tqdm import tqdm

#from PIL import Image
import numpy as np
import tensorflow as tf

import model

latent_size = 100
batch_size = 200
n_data = 1000

img_output_dir = 'output'
gan_checkpoint_path = os.path.join('checkpoint', 'final-1522487574.854423')

Z = tf.placeholder(tf.float32, [batch_size, latent_size], name='Z')
g_z = model.generator(Z, training=False)

with tf.name_scope('output'):
    Single_z = tf.placeholder(tf.float32, [784], name='Single_z')
    left, right = tf.reduce_min(Single_z), tf.reduce_max(Single_z)
    make_png = tf.subtract(Single_z, tf.tile([left], [784]))
    make_png = tf.scalar_mul(255. / (right - left), make_png)
    make_png = tf.cast(make_png, tf.uint8)
    make_png = tf.reshape(make_png, [28, 28, 1])
    make_png = tf.image.encode_png(make_png)

g_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
saver = tf.train.Saver(var_list=g_var)

with tf.Session() as sess:
    print()

    saver.restore(sess, gan_checkpoint_path)
    print('Model loaded\n')

    # Save images
    def save_images(imgs, name, offset):
        for i, img in enumerate(imgs):
            # Transform image
            png_encoded = sess.run(make_png, feed_dict={Single_z: img})

            file_path = os.path.join(
                    img_output_dir, '{}{:04d}.png'.format(name, i + offset))
            with open(file_path, 'wb') as f:
                f.write(png_encoded)

    for i in tqdm(range(0, n_data, batch_size)):
        Z_batch = np.random.normal(size=[batch_size, latent_size])
        imgs = sess.run(g_z, feed_dict={Z: Z_batch})

        save_images(imgs, 'img', i)

print('Done')
