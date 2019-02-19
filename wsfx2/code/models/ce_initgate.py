#将镜像门框架中的门机制改为参考论文中的门机制


import tensorflow as tf
import numpy as np


class modelConfig(object):
    def __init__(self):
        self.EMBDDING_DIM = 128
        self.FACT_LEN = 30
        self.LAW_LEN = 30
        self.KS_LEN = 3
        self.LAW_WIN = 8
        self.LAW_STRIDES = 1

        self.FILTERS = 256
        self.KERNEL_SIZE = 5  # 卷积核尺寸

        self.LAYER_UNITS = 100
        self.NUM_CLASS = 2

        self.LEARNING_RATE = 0.001
        self.batch_size = 128
        self.num_epochs = 200
        self.save_per_batch = 10
        self.print_per_batch = 10
        self.dropout_keep_prob = 0.5


class CNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x1 = tf.placeholder(tf.float32, [None, self.config.FACT_LEN, self.config.EMBDDING_DIM],
                                       name='input_x1')
        self.input_x2 = tf.placeholder(tf.float32, [None, self.config.LAW_LEN, self.config.EMBDDING_DIM],
                                       name='input_x2')
        self.input_ks = tf.placeholder(tf.float32, [None, self.config.KS_LEN, self.config.EMBDDING_DIM],
                                       name="input_ks")
        self.input_y = tf.placeholder(tf.int32, [None, self.config.NUM_CLASS],
                                      name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()
        return

    def cnn(self):
        self.new_ks = tf.reduce_mean(self.input_ks,axis=1,keep_dims=True)
        self.new_x1, pwls = self.gate(self.new_ks, self.input_x1)
        self.new_x1_mean = self.precessF2(self.new_x1)
        self.new_x1_mean_expd = tf.expand_dims(self.new_x1_mean,axis=1)
        self.new_x2 = self.gate(self.new_x1_mean_expd, self.input_x2)
        op1, op2 = self.conv(self.new_x1, self.new_x2)
        self.match(op1, op2)


    '''
    参考论文中门机制
    ks是一维向量
    '''
    def gate(self, ks, inputx):
        with tf.name_scope("gate"):
            weight_1 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM,self.config.EMBDDING_DIM],
                                                    stddev=0, seed=1),trainable=True, name='w1')
            weight_2 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, self.config.EMBDDING_DIM],
                                                    stddev=0, seed=1),trainable=True, name='w2')

            temp1 = tf.einsum('abc,ce->abe',inputx,weight_1) #[b,l,d]
            temp2 = tf.keras.backend.repeat_elements(tf.einsum('abc,ce->abe',ks,weight_2),rep=inputx.shape[1],axis=1) #[b,l,d]


            pw = tf.sigmoid(temp1+temp2)

            one_array = tf.ones(shape=pw.shape)

            new_vector = (one_array - pw) * inputx + pw * ks
        return new_vector


    def precessF2(self, inputx):
        with tf.name_scope("FactPrecess"):
            inputx_ = tf.transpose(inputx, perm=[0, 2, 1])
            inputx_k = tf.transpose((tf.nn.top_k(inputx_, k=5,   #***************check!!!!
                                                 sorted=False))[0], perm=[0, 2, 1])  # [None,k,d]
            inputx_mean = tf.reduce_mean(inputx_k, axis=1)
            return inputx_mean



    def conv(self, inputx, inputy):
        with tf.name_scope("conv"):
            conv1 = tf.layers.conv1d(inputx, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                                     name='conv1')
            op1 = tf.reduce_max(conv1, reduction_indices=[1], name='gmp1')

            conv2 = tf.layers.conv1d(inputy, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                                     name='conv2')
            op2 = tf.reduce_max(conv2, reduction_indices=[1], name='gmp2')

            return op1, op2

    def match(self, op1, op2):
        with tf.name_scope("match"):
            h = tf.concat([op1, op2], axis=1)  # [batch,FILTERS*2] #***************check!!!!
            fc = tf.layers.dense(inputs=h, units=self.config.LAYER_UNITS, use_bias=True,
                                 trainable=True, name="fc1")
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)  # 根据比例keep_prob输出输入数据，最终返回一个张量
            fc = tf.nn.relu(fc)  # 激活函数，此时fc的维度是hidden_dim

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.NUM_CLASS,
                                          name='fc2')  # 将fc从[batch_size,hidden_dim]映射到[batch_size,num_class]输出
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.input_y)  # 对logits进行softmax操作后，做交叉墒，输出的是一个向量
            # regularizer = tf.contrib.layers.l2_regularizer(scale=5.0 / 50000)
            # reg_term = tf.contrib.layers.apply_regularization(regularizer)
            self.loss = tf.reduce_mean(cross_entropy)  # 将交叉熵向量求和，即可得到交叉熵
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.LEARNING_RATE).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),
                                    self.y_pred_cls)  # 由于input_y也是onehot编码，因此，调用tf.argmax(self.input_y)得到的是1所在的下表
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
