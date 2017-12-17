from __future__  import print_function
import tensorflow as tf
import numpy as np
import time

from utils import process_data, visualise, CosineTest

class w2v(object):

    def __init__(self):
        self.BatchSize = 32
        self.VocabSize = 50000
        self.FeatSize = 600
        self.NumSampled = 64
        self.lr = 0.08
        self.NumTrainSteps = 200000
        self.ReportStep = 1000
        self.EmbedVectorFile = 'data/embed_vectors.txt'
        self.VisualizeWordFile = 'data/Visualwords.txt'

    def _create_placeholders(self):
        with tf.name_scope('data') as scope:
            self.center_words = tf.placeholder(dtype='int32', shape=[self.BatchSize])
            self.target_words = tf.placeholder(dtype='int32', shape=[self.BatchSize, 1])

    def _create_embedding(self):
        with tf.name_scope('embedding') as scope:
            self.embed_weights = tf.get_variable(initializer=tf.random_uniform_initializer((-1.0, 1.0)),\
                                                 shape=[self.VocabSize, self.FeatSize])

    def _create_loss(self):
        with tf.name_scope('loss') as scope:
            embed = tf.nn.embedding_lookup(self.embed_weights, self.center_words)

            nce_weights = tf.get_variable(initializer=tf.random_uniform_initializer((-1.0, 1.0)),\
                                          shape=[self.VocabSize, self.FeatSize])
            nce_bias = tf.get_variable(initializer=tf.random_uniform_initializer((-1.0, 1.0)), shape=[self.VocabSize])

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_bias,\
                                                      labels=self.target_words, inputs=embed,\
                                                      num_classes=self.VocabSize, num_sampled=self.NumSampled))

    def _create_optimizer(self):
        self.update_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


def train_w2v(model, batch_gen):
    model._create_placeholders()
    model._create_embedding()
    model._create_loss()
    model._create_optimizer()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

#
#
#
#
#
# BatchSize = 32
# VocabSize = 50000
# FeatSize = 600
# NumSampled = 64
# WindowSize = 7
# LearningRate = 0.08
# NumTrainSteps = 200000
# ReportStep = 1000
# EmbedVectorFile = 'data/embed_vectors.txt'
# VisualizeWordFile = 'data/Visualwords.txt'
#
# with tf.name_scope("data") as scope:
#     center_words = tf.placeholder(dtype='int32', shape=[BatchSize])
#     target_words = tf.placeholder(dtype='int32', shape=[BatchSize, 1])
#
# with tf.device('/cpu:0'):
#     with tf.name_scope('embeddings')as scope:
#         embed_weights = tf.Variable(tf.random_uniform([VocabSize, FeatSize], -1.0, 1.0),\
#                                     name='weights')
#
#         embeds = tf.nn.embedding_lookup(embed_weights, center_words)
#
#     with tf.name_scope('loss') as scope:
#         nce_weights = tf.Variable(tf.truncated_normal([VocabSize, FeatSize]), dtype='float32',\
#                                   name='soft_weights')
#
#         nce_bias = tf.Variable(tf.zeros([VocabSize]), dtype='float32', name='soft_bias')
#
#         losses = tf.nn.nce_loss(weights=nce_weights, biases=nce_bias, inputs=embeds,\
#                                 labels=target_words, num_sampled=NumSampled, num_classes=VocabSize, name='loss')
#
#         loss = tf.reduce_mean(losses)
#     update_op = tf.train.AdagradOptimizer(LearningRate).minimize(loss)
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     data_gen = process_data(VocabSize, BatchSize, WindowSize)
#     np.savetxt('data/initial_weights.txt', embed_weights.eval())
#     for step in range(NumTrainSteps):
#         centers, targets = next(data_gen)
#         loss_np, _ = sess.run([loss, update_op], feed_dict={center_words: centers, target_words: targets})
#         if step%ReportStep == 0:
#             # print("len of center words", centers.shape)
#             # print("len of targets words", targets.shape)
#             # print("center[:10]\n", centers[:10])
#             # print("target[:10]\n", targets[:10])
#             print("at step {0} loss is {1}".format(step, loss_np))
#     print("writing EmbedVector at", EmbedVectorFile)
#     time.sleep(5)
#     print("saving weights")
#     np.savetxt(EmbedVectorFile, embed_weights.eval())
#     time.sleep(5)
#     print('saving png')
#     print(CosineTest(VisualizeWordFile, EmbedVectorFile, querry='american'))
