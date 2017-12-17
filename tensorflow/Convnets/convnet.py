import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)

x = tf.placeholder(dtype='float32', shape=[None, 784], name='X')
y = tf.placeholder(dtype='float32', shape=[None, 10], name='Y')

with tf.variable_scope('conv1') as scope:
    image_variable = tf.reshape(x, [-1, 28, 28, 1])
    kernel = tf.get_variable(name='weights', shape=[3, 3, 1, 32], initializer=tf.truncated_normal_initializer(),\
                             dtype='float32')
    b = tf.get_variable(name='bias', dtype='float32', initializer=tf.truncated_normal_initializer(), shape=[32])
    feat = tf.nn.conv2d(image_variable, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.add(tf.nn.relu(feat), b, name='conv')

with tf.variable_scope('pool1') as scope:
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool', padding='SAME')

with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable(dtype='float32', shape=[3, 3, 32, 32], initializer=tf.truncated_normal_initializer(),\
                             name='weights')
    b = tf.get_variable(dtype='float32', shape=[32], initializer=tf.truncated_normal_initializer(), name='bias')
    feat = tf.nn.conv2d(pool, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.add(tf.nn.relu(feat), b, name='conv')

with tf.variable_scope('pool2') as scope:
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool', padding='SAME')

with tf.variable_scope('conv3') as scope:
    kernel = tf.get_variable(dtype='float32', shape=[3, 3, 32, 64], initializer=tf.truncated_normal_initializer(),\
                             name='weigthts')
    b = tf.get_variable(dtype='float32', shape=[64], initializer=tf.truncated_normal_initializer(), name='bias')
    feat = tf.nn.conv2d(pool, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.add(tf.nn.relu(feat), b, name='conv')

with tf.variable_scope('FC') as scope:
    conv2 = tf.reshape(conv, [-1, 3136])
    w = tf.get_variable(dtype='float32', shape=[3136, 1024], initializer=tf.truncated_normal_initializer(),\
                              name='weights')
    b = tf.get_variable(dtype='float32', shape=[1024], initializer=tf.truncated_normal_initializer(), name='bias')
    feat = tf.matmul(conv2, w) + b

with tf.variable_scope('FC2') as scope:
    w = tf.get_variable(dtype='float32', shape=[1024, 10], initializer=tf.truncated_normal_initializer(),
                        name='weights')
    b = tf.get_variable(dtype='float32', shape=[10], initializer=tf.truncated_normal_initializer(), name='bias')
    logits = tf.matmul(feat, w) + b

with tf.variable_scope('softmax_logits') as scope:
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(entropy)

update_op = tf.train.AdagradOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps = int(mnist.train.num_examples/50.0)
    print("steps are", steps)
    for i in range(1):
        losses =[]
        for step in range(steps):
            X, Y = mnist.train.next_batch(50)
            _, loss_np = sess.run([update_op, loss], feed_dict={x: X, y: Y})
            losses.append(loss_np)
            # print("step=", step, " loss=", loss_np/50.0)
        print("at iteration =", i, " loss =", float(sum(losses))/len(losses))

    # test network
    steps = int(mnist.test.num_examples)
    predictions = []
    print("test steps are ", steps)
    for step in range(steps):
        X, Y = mnist.train.next_batch(50)
        logits_np = sess.run([logits], feed_dict={x: X.astype('float32'), y: Y.astype('float32')})
        print("prediction ", np.shape(sess.run(tf.nn.softmax(logits=logits_np[0]))))
        predict = np.argmax(sess.run(tf.nn.softmax(logits=logits_np[0])))
        print('gt', Y)
        gt = np.argmax(Y)
        if predict == gt:
            predictions.append(1.0)
        else:
            predictions.append(0.0)
    print('accuracy =', float(sum(predictions))/len(predictions))