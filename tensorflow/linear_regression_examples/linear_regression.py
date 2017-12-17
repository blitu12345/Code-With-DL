import tensorflow as tf
import pandas as pd
import numpy as np

sh = pd.ExcelFile('fire_theft.xls')
df = sh.parse('Sheet 1')

X = df.X.values
Y = df.Y.values

x = tf.placeholder('float32', name='X')
y = tf.placeholder('float32', name='Y')

w = tf.Variable(tf.random_uniform((1, 1)), name='w')
b = tf.Variable(tf.random_uniform((1, 1)), name='b')

y_pred = tf.add(tf.matmul(x, w), b, name="y_pred")
error = tf.subtract(y_pred, y, name='error')
tf.summary.scalar('error', tf.reduce_sum(error, 0)[0])
lr = tf.constant(0.001)
grad = tf.reduce_sum(x*2*error, 0, name='gradient')

update_op = tf.assign(w, tf.subtract(w, lr*grad/len(X)), name='update_weights')
update_op2 = tf.assign(b, tf.subtract(b, lr*tf.reduce_sum(2*error, 0)/len(X)), name='update_bias')

summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('graph_data/', graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    epochs = 10000
    for epoch in range(epochs):
        _, _, summary_str, error_np = sess.run([update_op, update_op2, summary_op, error], \
                                               feed_dict={x: X.reshape(len(X), 1), y: Y.reshape(len(Y), 1)})
        print("sumed error={} at {}epoch".format(sum(error_np), epoch))
        if epoch % 10 == 0:
            summary_writer.add_summary(summary_str, epoch)
            summary_writer.flush()
