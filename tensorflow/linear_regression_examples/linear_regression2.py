import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sh = pd.ExcelFile('fire_theft.xls')
df = sh.parse('Sheet 1')

X = df.X.values
Y = df.Y.values

with tf.name_scope('input'):
    x = tf.placeholder('float32', name='in_X')
    y = tf.placeholder('float32', name='in_Y')

with tf.name_scope('weights'):
    w = tf.Variable(tf.random_uniform((1, 1)), name='weights')

with tf.name_scope('bias'):
    b = tf.Variable(tf.random_uniform((1, 1)), name='bias')

with tf.name_scope('prediction'):
    y_pred = tf.matmul(x, w) + b
    y_pred = tf.identity(y_pred, name='y_pred')

with tf.name_scope('error'):
    error = tf.subtract(y_pred, y, name='error_cal')

with tf.name_scope('update'):
    lr = tf.constant(0.10, name='learning_rate')
    average_weight_gradient = tf.reduce_sum(2*error*x, 0, name='average_weight_gradient')/len(X)
    update_op1 = tf.assign(w, tf.subtract(w, lr*average_weight_gradient/len(X)), name='update_weights')

    average_bias_gradient = tf.reduce_sum(2*error, 0, name='average_bias_gradient')/len(X)
    update_op2 = tf.assign(b, tf.subtract(b, lr*average_bias_gradient/len(X)), name='update_bias')

tf.summary.scalar('error', tf.reduce_sum(error, 0)[0])
summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('graph_data/', graph=sess.graph)
    epochs = 10000
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        _, _, summary_str, error_np = sess.run([update_op1, update_op2, summary_op, error], feed_dict={x: X.reshape(len(X), 1), y: Y.reshape(len(Y), 1)})
        print "epoch {} error {}".format(epoch, sum(error_np))
        if epoch % 10 == 0:
            summary_writer.add_summary(summary_str, epoch)
            summary_writer.flush()

    w, b = sess.run([w, b])
    plt.scatter(X, Y, color='b', label='real_data')
    plt.plot(X, (X * w + b).reshape(len(X), ), color='r', label='predicted_line')
    plt.legend()
    plt.show()