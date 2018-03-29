import tensorflow as tf

conv = tf.layers.conv2d
fc = tf.layers.dense
bnorm = tf.layers.batch_normalization
dropout = tf.layers.dropout

relu = tf.nn.relu
relu = tf.nn.leaky_relu
sigmoid = tf.sigmoid
tanh = tf.tanh

def generator(Z, training):
    with tf.variable_scope('generator'):
        fc1 = fc(Z, 256, name='hidden1')
        fc1 = relu(fc1)

        fc2 = fc(fc1, 512, name='hidden2')
        fc2 = relu(fc2)

        fc3 = fc(fc2, 1024, name='hidden3')
        fc3 = relu(fc3)

        output = fc(fc3, 784, name='output')
        output = tanh(output)
    return output

def discriminator(X, training, fake=False):
    with tf.variable_scope('discriminator', reuse=fake):
        fc1 = fc(X, 1024, name='hidden1')
        fc1 = relu(fc1)
        fc1 = dropout(fc1, 0.3)

        fc2 = fc(fc1, 512, name='hidden2')
        fc2 = relu(fc2)
        fc2 = dropout(fc2, 0.3)

        fc3 = fc(fc2, 256, name='hidden3')
        fc3 = relu(fc3)
        fc3 = dropout(fc3, 0.3)

        output = fc(fc3, 1, name='output')
        output = sigmoid(output)
    return output
