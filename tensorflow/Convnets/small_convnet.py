import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)

x = tf.placeholder(dtype='float32', name='x')
y = tf.placeholder(dtype='float32', name='y', shape=[None, 10])

# assert x.shape == [50, 784]

with tf.variable_scope('conv') as scope:
    # image_tensor = tf.reshape(x, [-1, 28, 28, 1])
    kernel = tf.get_variable(name='weights', dtype='float32', shape=[5, 5, 1, 32],\
                             initializer=tf.truncated_normal_initializer())
    b = tf.get_variable(name='bias', dtype='float32', shape=[32], initializer=tf.truncated_normal_initializer())
    feat = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.relu(name='relu', features=feat + b)

# assert conv.get_shape() == [50, 28, 28, 32]

with tf.variable_scope('pool') as scope:
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# assert pool.get_shape() == [50, 14, 14, 32]

with tf.variable_scope('FC') as scope:
    pool_reshape = tf.reshape(pool, [-1, 6272])
    w = tf.get_variable(name='weights', dtype='float32', shape=[6272, 1024], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable(name='bias', dtype='float32', shape=[1024], initializer=tf.truncated_normal_initializer())
    fc = tf.add(tf.matmul(pool_reshape, w), b, name='FC')

# assert fc.get_shape() == [50, 1024]

with tf.variable_scope('softmax') as scope:
    w = tf.get_variable(name='weights', dtype='float32', shape=[1024, 10], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable(name='bias', dtype='float32', shape=[10], initializer=tf.truncated_normal_initializer())
    logits = tf.add(tf.matmul(fc, w), b, name='logits')

with tf.variable_scope('loss') as scope:
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_sum(entropy, name='loss')

update_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        steps = int(mnist.train.num_examples / 50)
        for step in range(steps):
            X_batch, Y_batch = mnist.train.next_batch(50)
            _, loss_np = sess.run([update_op, loss], feed_dict={x: X_batch.reshape((-1, 28, 28, 1)), y: Y_batch})
            print("step", step, "loss", loss_np)
