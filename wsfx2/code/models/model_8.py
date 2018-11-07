#加入上下文的gate2 in model5

# 添加多级先验知识，并且上一层级得到的[1，d]的score会传入下一层的下一级的计算中，使用的每层计算的权重是两个[d,1]的


import tensorflow as tf
import numpy as np


class modelConfig(object):
    def __init__(self):
        self.EMBDDING_DIM = 128
        self.FACT_LEN = 30
        self.LAW_LEN = 30
        self.KS_LEN = 3

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
        self.input_x1 = tf.placeholder(tf.float32, [None, self.config.FACT_LEN , self.config.EMBDDING_DIM],
                                       name='input_x1')
        self.input_x2 = tf.placeholder(tf.float32, [None, self.config.LAW_LEN , self.config.EMBDDING_DIM],
                                       name='input_x2')
        self.input_ks = tf.placeholder(tf.float32, [None, self.config.KS_LEN, self.config.EMBDDING_DIM],
                                       name="input_ks")
        self.input_y = tf.placeholder(tf.int32, [None, self.config.NUM_CLASS],
                                      name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()
        return

    def cnn(self):
        # context = self.count_context(self.input_x1)
        self.new_x1, pwls = self.gate1(self.input_ks, self.input_x1)
        self.new_x2 = self.gate1_s2l(self.new_x1, self.input_x2)
        op1, op2 = self.conv(self.new_x1, self.new_x2)
        self.match(op1, op2)

    '''
    s1 = tf.sigmoid(tf.relu(x * w * k_1)) 计算得到关于这个先验知识哪些dim应该被保留,这个weight是[128,1]
    s2 = tf.sigmoid(tf.relu(x * w * [k_2;s1]))
    s3 = tf.sigmoid(tf.relu(x * w * [k_3;s2]))
    new_vector = s1 * x + s2 * x + s3 * x
    '''

    def gate1(self, ks, inputx):
        with tf.name_scope("gate"):
            weight_1 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, 1],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            weight_2 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, 2],
                                                    stddev=0, seed=2), trainable=True, name='w2')
            weight_3 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, 2],
                                                    stddev=0, seed=3), trainable=True, name='w3')

            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight_1)
            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight_1)
            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight_3)

            k_1_init, k_2_init, k_3_init = ks[:, 0, :], ks[:, 1, :], ks[:, 2, :]  # [None,d]
            k_1 = tf.reshape(tf.keras.backend.repeat_elements(k_1_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBDDING_DIM])
            k_2 = tf.reshape(tf.keras.backend.repeat_elements(k_2_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBDDING_DIM])
            k_3 = tf.reshape(tf.keras.backend.repeat_elements(k_3_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBDDING_DIM])
            inputx_epd = tf.expand_dims(inputx, axis=2) #[b,l,1,d]
            fun1 = tf.einsum('abcd,de->abce', inputx_epd, weight_1)
            ksw_1 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun1, k_2)))  # [batch,l,1,d]

            fun2 = tf.einsum('abcd,de->abce', inputx_epd, weight_2)
            ksw_2 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun2, tf.concat([k_3, ksw_1], axis=2))))  # [batch,l,d]

            fun3 = tf.einsum('abcd,de->abce',inputx_epd , weight_3)
            ksw_3 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun3, tf.concat([k_1, ksw_2], axis=2))))  # [batch,l,d]

            n_vector_ = (ksw_1 + ksw_2 + ksw_3) * inputx_epd
            n_vector = tf.reshape(n_vector_, shape=[-1,self.config.FACT_LEN,self.config.EMBDDING_DIM])

        return n_vector, tf.concat([ksw_1, ksw_2, ksw_3], axis=2)

        '''
        基于gate1，加上前一个词和后一个词的每个级别的概率值作为先验知识的一部分
        s1 = tf.sigmoid(tf.relu([xf;xh] * w * [k_1])) 计算得到关于这个先验知识哪些dim应该被保留,这个weight是[128,1]
        s2 = tf.sigmoid(tf.relu([xf;xh] * [k_2;s1]))
        s3 = tf.sigmoid(tf.relu([xf;xh] * w * [k_3;s2]))
        new_vector = s1 * x + s2 * x + s3 * x

        '''
    def count_context(self,inputx):
        with tf.name_scope("cmp_context"):
            inputx_re = tf.reshape(inputx,shape=[self.config.FACT_LEN+2,-1])
            new_inputx = []
            for i in range(1,self.config.FACT_LEN+1):
                mean = tf.add(inputx_re[i-1,:] , inputx_re[i+1,:])
                mean = tf.convert_to_tensor(mean)
                new_inputx.append(mean)

            nputx_ctx = tf.convert_to_tensor(new_inputx)
            nputx_ctx = tf.reshape(nputx_ctx,shape=[-1,self.config.FACT_LEN,self.config.EMBDDING_DIM])
            return nputx_ctx

    #使用上下文的
    def gate2(self, ks, inputx, nputx_ctx):
        with tf.name_scope("gate"):
            weight_1 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, 1],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            weight_2 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, 2],
                                                    stddev=0, seed=2), trainable=True, name='w2')
            weight_3 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, 2],
                                                    stddev=0, seed=3), trainable=True, name='w3')

            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight_1)
            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight_1)
            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight_3)

            k_1_init, k_2_init, k_3_init = ks[:, 0, :], ks[:, 1, :], ks[:, 2, :]  # [None,d]
            k_1 = tf.reshape(tf.keras.backend.repeat_elements(k_1_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBDDING_DIM])
            k_2 = tf.reshape(tf.keras.backend.repeat_elements(k_2_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBDDING_DIM])
            k_3 = tf.reshape(tf.keras.backend.repeat_elements(k_3_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBDDING_DIM])

            nputx_ctx_epd = tf.expand_dims(nputx_ctx,axis=2) #[None,l,1,d]
            inputx_epd = tf.expand_dims(inputx[:,1:self.config.FACT_LEN+1,:],axis=2) #[None,l,1,d]

            fun1 = tf.einsum('abcd,de->abce',nputx_ctx_epd , weight_1)
            ksw_1 = tf.sigmoid(
                tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun1, k_1)))  # [batch,l,1,d]

            fun2 = tf.einsum('abcd,de->abce',nputx_ctx_epd , weight_2)
            ksw_2 = tf.sigmoid(
                tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun2, tf.concat([k_2, ksw_1], axis=2))))  # [batch,l,d]

            fun3 = tf.einsum('abcd,de->abce',nputx_ctx_epd , weight_3)
            ksw_3 = tf.sigmoid(
                tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun3, tf.concat([k_3, ksw_2], axis=2))))  # [batch,l,d]

            n_vector_ = (ksw_1 + ksw_2 + ksw_3) * inputx_epd
            n_vector = tf.reshape(n_vector_, shape=[-1, self.config.FACT_LEN, self.config.EMBDDING_DIM])

        return n_vector, tf.concat([ksw_1, ksw_2, ksw_3], axis=2)

    #comparision experiment of gate1
    def gate3(self, ks, inputx):
        with tf.name_scope("gate"):
            weight_1 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, 1],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            weight_2 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, 1],
                                                    stddev=0, seed=2), trainable=True, name='w2')
            weight_3 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, 1],
                                                    stddev=0, seed=3), trainable=True, name='w3')

            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight_1)
            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight_1)
            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight_3)

            k_1_init, k_2_init, k_3_init = ks[:, 0, :], ks[:, 1, :], ks[:, 2, :]  # [None,d]
            k_1 = tf.reshape(tf.keras.backend.repeat_elements(k_1_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBDDING_DIM])
            k_2 = tf.reshape(tf.keras.backend.repeat_elements(k_2_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBDDING_DIM])
            k_3 = tf.reshape(tf.keras.backend.repeat_elements(k_3_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, 1, self.config.EMBDDING_DIM])
            inputx_epd = tf.expand_dims(inputx, axis=2) #[b,l,1,d]
            fun1 = tf.einsum('abcd,de->abce', inputx_epd, weight_1)
            ksw_1 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun1, k_1)))  # [batch,l,1,d]

            fun2 = tf.einsum('abcd,de->abce', inputx_epd, weight_2)
            ksw_2 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun2, k_2)))  # [batch,l,d]

            fun3 = tf.einsum('abcd,de->abce',inputx_epd , weight_3)
            ksw_3 = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,abdf->abcf', fun3, k_3)))  # [batch,l,d]

            n_vector_ = (ksw_1 + ksw_2 + ksw_3) * inputx_epd
            n_vector = tf.reshape(n_vector_, shape=[-1,self.config.FACT_LEN,self.config.EMBDDING_DIM])

        return n_vector, tf.concat([ksw_1, ksw_2, ksw_3], axis=2)

    '''
    根据事实作为先验知识去过滤法条
    '''
    def gate1_s2l(self,inputx,inputy):
        with tf.name_scope("Fact2Law"):
            weight_1 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, 1],
                                                    stddev=0, seed=1), trainable=True, name='w1')
            ss_epd = tf.expand_dims(inputx, axis=2)  # [b,l,1,d]
            law_epd = tf.expand_dims(inputy,axis=2) #[b,l,1,d]
            fun = tf.einsum('abcd,de->abce', law_epd, weight_1)
            ksw = tf.sigmoid(tf.nn.relu(tf.einsum('abcd,de->abce'),fun, ss_epd)) #[None,l,1,d]

            n_vector_ = ksw * law_epd
            n_vector = tf.reshape(n_vector_, shape=[-1,self.config.LAW_LEN,self.config.EMBDDING_DIM])
        return n_vector

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
            h = tf.concat([op1, op2], axis=1)  # [batch,FILTERS*2]
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
