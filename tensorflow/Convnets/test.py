import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)

input_size = 784
epochs = 100
batch_size = 50
lr = 0.001

X = tf.placeholder(dtype='float32', name='x')
Y = tf.placeholder(dtype='float32', name='y')

with tf.variable_scope('conv1') as scope:
    w = tf.get_variable('weights', [3, 3, 1, 32], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('bias', [32], initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv + b, name=scope.name)

with tf.variable_scope('pool1') as scope:
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope('conv2') as scope:
    w = tf.get_variable('weights', [3, 3, 32, 32], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('bias', [32], initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(pool1, w, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv + b, name=scope.name)

with tf.variable_scope('pool2') as scope:
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.variable_scope('conv3') as scope:
    w = tf.get_variable('weights', [3, 3, 32, 64], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('bias', [64], initializer=tf.random_uniform_initializer())
    conv = tf.nn.conv2d(pool2, w, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(conv + b, name=scope.name)

with tf.variable_scope('FC1') as scope:

    w = tf.get_variable('weights', [7*7*64, 1024], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('bias', [1024], initializer=tf.random_uniform_initializer())
    flatten_conv3 = tf.reshape(conv3, [-1, 7*7*64])

    fc1 = tf.nn.relu(tf.matmul(flatten_conv3, w) + b, name='relu')
    fc1 = tf.nn.dropout(fc1, 0.25, name='dropout')

with tf.variable_scope('softmax_linear') as scope:
    w1 = tf.get_variable('weights', [1024, 10], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable('bias', [10], initializer=tf.random_uniform_initializer())
    fc2 = tf.matmul(fc1, w1) + b1

with tf.variable_scope('softmax_and_loss') as scope:
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=Y)
    loss = tf.reduce_mean(tf.clip_by_value(entropy, 1e-10, 1.0), name='loss')

update_op = tf.train.AdamOptimizer(0.05).minimize(loss)


tf.summary.scalar('error', loss)
summary_ = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('graph_data/', graph=sess.graph)
    sess.run(tf.global_variables_initializer())

    steps = mnist.train.num_examples/batch_size

    for epoch in range(epochs):
        avg_error = []
        for step in range(steps):
            X_, Y_ = mnist.train.next_batch(batch_size)
            _, error, summary_str, w_np, b_np = sess.run([update_op, loss, summary_, w, b],\
                                                         feed_dict={X: np.reshape(X_, (batch_size, 28, 28, 1)), Y: Y_})
            avg_error.append(error)
            if step % 10 == 0:
                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()
        print("error {} at epoch{}".format(sum(avg_error)/(batch_size*steps), epoch))
        print("w", w_np[1])
        print("b", b_np.shape)