import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST_data", one_hot=True)

batch_size = 50

with tf.name_scope('input'):
    x = tf.placeholder(dtype='float32', name='input_x')
    y = tf.placeholder(dtype='float32', name='input_y')

with tf.name_scope('weight'):
    w = tf.Variable(tf.random_uniform((784, 10)), dtype='float32', name='weights')

with tf.name_scope('bias'):
    b = tf.Variable(tf.random_uniform((1, 10)), dtype='float32', name='bias')

with tf.name_scope('predict'):
    y_pred = tf.matmul(x, w) + b

with tf.name_scope('loss'):
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y)
    loss = tf.reduce_mean(entropy, axis=0)

with tf.name_scope('backprop_update'):
    update_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.name_scope('test'):
    y_class = tf.nn.softmax(y_pred)
    # cond = tf.equal(y_class, y)

tf.summary.scalar('loss', loss)
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps = mnist.train.num_examples/batch_size
    epochs = 50
    for epoch in range(epochs):
        error_avg = []
        for step in range(steps):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            # print(np.shape(X_batch), np.shape(Y_batch)) # ((50, 784), (50, 10))
            _, error = sess.run([update_op, loss], feed_dict={x: X_batch, y: Y_batch})
            error_avg.append(error)
            # print("error={} at step={}".format(error/50.0, steps))
        print("at epoch={} error={}".format(epoch, sum(error_avg)/(mnist.train.num_examples)))

    print "weights\n", sess.run([w])

    steps = mnist.test.num_examples
    accuracy = 0.0
    correct_count = 0
    for step in range(steps):
        X, Y = mnist.test.next_batch(1)
        y_class_np = np.asarray(sess.run([y_class], feed_dict={x: X})).reshape(10, )
        Y = Y.reshape(10,)
        # print "Y_pred",y_class_np, np.shape(y_class_np)
        # print "y", Y, np.shape(Y)
        # print "np.where(y_class_np == 1.0) ", np.argmax(y_class_np)
        # print "np.where(Y == 1.0)", np.where(Y == 1.0)[0]
        if np.argmax(y_class_np) == np.where(Y == 1.0)[0]:
            correct_count += 1.0
    print "accuracy={}".format(correct_count/steps)
