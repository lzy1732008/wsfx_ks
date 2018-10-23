"""This is the Model File of MatchPyramid.

This module is used to construct the MatchPyramid described in paper https://arxiv.org/abs/1602.06359.
"""

import sys

import tensorflow as tf
import numpy as np

"""
Model Class
"""
class modelconfig():
    def __init__(self):
        self.data1_maxlen = 30
        self.data2_maxlen = 50
        self.embedding = 50
        self.data1_psize = 3
        self.data2_psize = 5
        self.LEARNING_RATE = 0.001
        self.batch_size = 128
        self.dropout_keep_prob = 0.5


class MatchPy():

    def __init__(self, config):
        self.config = config
        tf.reset_default_graph()
        self.X1 = tf.placeholder(tf.int32, name='X1', shape=(None, self.config.data1_maxlen))
        self.X2 = tf.placeholder(tf.int32, name='X2', shape=(None, self.config.data2_maxlen))
        self.X1_len = tf.placeholder(tf.int32, name='X1_len', shape=(None,))
        self.X2_len = tf.placeholder(tf.int32, name='X2_len', shape=(None,))
        self.Y = tf.placeholder(tf.int32, name='Y',shape=(None,2))
        self.keep_prob = tf.placeholder(tf.int32, name='keep_prob',shape=(None,))


        self.dpool_index = self.dynamic_pooling_index(self.X1_len, self.X2_len,
                                                      self.config.data1_maxlen,self.confi.data2_maxlen)

        self.batch_size = tf.shape(self.X1)[0]

        self.embedding = tf.get_variable('embedding', initializer=self.config.embedding, dtype=tf.float32,
                                         trainable=False)

        self.embed1 = tf.nn.embedding_lookup(self.embedding, self.X1)
        self.embed2 = tf.nn.embedding_lookup(self.embedding, self.X2)

        # batch_size * X1_maxlen * X2_maxlen
        self.cross = tf.einsum('abd,acd->abc', self.embed1, self.embed2)
        self.cross_img = tf.expand_dims(self.cross, 3)

        # convolution
        self.w1 = tf.get_variable('w1',
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2, dtype=tf.float32),
                                  dtype=tf.float32, shape=[2, 10, 1, 8])
        self.b1 = tf.get_variable('b1', initializer=tf.constant_initializer(), dtype=tf.float32, shape=[8])
        # batch_size * X1_maxlen * X2_maxlen * feat_out
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.cross_img, self.w1, [1, 1, 1, 1], "SAME") + self.b1)

        # dynamic pooling
        self.conv1_expand = tf.gather_nd(self.conv1, self.dpool_index)
        self.pool1 = tf.nn.max_pool(self.conv1_expand,
                                    [1, self.config.data1_maxlen / self.config.data1_psize,
                                     self.config.data2_maxlen / self.config.data2_psize, 1],
                                    [1, self.config.data1_maxlen / self.config.data1_psize,
                                     self.config.data2_maxlen / self.config.data2_psize, 1], "VALID")

        with tf.variable_scope('fc'):
            fc = tf.layers.dense(inputs=tf.reshape(self.pool1, [self.batch_size, self.config.data1_psize * self.config.data2_psize * 8]),
                                       units=20,use_bias=True,trainable=True,name="fc1")
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            self.logits = tf.layers.dense(fc, 2)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.Y)  # 对logits进行softmax操作后，做交叉墒，输出的是一个向量
            self.loss = tf.reduce_mean(cross_entropy)  # 将交叉熵向量求和，即可得到交叉熵
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.LEARNING_RATE).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.Y, 1),self.y_pred_cls)  # 由于input_y也是onehot编码，因此，调用tf.argmax(self.input_y)得到的是1所在的下表
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i / stride1) for i in range(max_len1)]
            idx2_one = [int(i / stride2) for i in range(max_len2)]
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2, 1, 0))
            return index_one

        index = []
        for i in range(len(len1)):
            index.append(dpool_index_(i, len1[i], len2[i], max_len1, max_len2))
        return np.array(index)


