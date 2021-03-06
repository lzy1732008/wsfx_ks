#添加多级先验知识,gate is differ from model_3.py

#基于CNN，只使用最精细的那个先验知识，conv1d

import tensorflow as tf


class modelConfig(object):
    def __init__(self):
        self.EMBDDING_DIM = 128
        self.FACT_LEN = 30
        self.LAW_LEN = 30
        self.KS_LEN= 3

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
        new_x1 = self.gate(self.input_ks,self.input_x1)
        op1,op2 = self.conv(new_x1,self.input_x2)
        self.match(op1,op2)



    def gate(self, ks, inputx):#pw = sigmoid([ks;e]*w*e)->[4,d]代表各个的概率,new_vector = pw[0] * ks[0] + pw[1] * ks[1] + pw[2] * ks[2] + pw[3] * e
        with tf.name_scope("gate"):
            weight_1 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM,self.config.EMBDDING_DIM],
                                                    stddev=0, seed=1),trainable=True, name='w1')
            weight_2 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, self.config.EMBDDING_DIM],
                                                    stddev=0, seed=1),trainable=True, name='w2')
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight_1)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, weight_1)

            ksw = tf.einsum('abc,cd->abd', ks, weight_1) #[None, 3, d]
            ew = tf.einsum('abc,cd->abd', inputx, weight_2) #[None, l, d]

            print('ksw.shape',ksw.shape)
            ksw_mul_pre = tf.keras.backend.repeat_elements(ksw, rep=self.config.FACT_LEN, axis = 1)
            print('ksw_mul_pre.type', ksw_mul_pre)
            ksw_mul = tf.reshape(ksw_mul_pre, shape=[-1, self.config.FACT_LEN, self.config.KS_LEN,self.config.EMBDDING_DIM])
            print('ksw_mul.shape', ksw_mul.shape)
            ew_mul = tf.expand_dims(ew, axis=2)

            temp1 = tf.concat([ksw_mul,ew_mul], axis=2)#[None, l, 4, d]
            pw = tf.sigmoid(temp1) #[None, l, 4, d]
            new_vector1 = tf.reduce_sum(temp1 * pw, axis=2) #[None, l, 1, d]
            new_vector = tf.reshape(new_vector1, shape=[-1,self.config.FACT_LEN,self.config.EMBDDING_DIM])

        return new_vector



    def conv(self,inputx,inputy):
        with tf.name_scope("conv"):
            conv1 = tf.layers.conv1d(inputx,filters=self.config.FILTERS,kernel_size=self.config.KERNEL_SIZE,name='conv1')
            op1 = tf.reduce_max(conv1, reduction_indices=[1], name='gmp1') #[None,256]

            conv2 = tf.layers.conv1d(inputy,filters=self.config.FILTERS,kernel_size=self.config.KERNEL_SIZE,name='conv2')
            op2 = tf.reduce_max(conv2, reduction_indices=[1], name='gmp2') #[None,256]

            return op1,op2

    def match(self,op1,op2):
        with tf.name_scope("match"):

            h = tf.concat([op1,op2],axis=1) #[batch,len1+len2]
            fc = tf.layers.dense(inputs=h, units= self.config.LAYER_UNITS, use_bias=True,
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
            regularizer = tf.contrib.layers.l2_regularizer(scale=5.0 / 50000)
            reg_term = tf.contrib.layers.apply_regularization(regularizer)
            self.loss = tf.reduce_mean(cross_entropy + reg_term)  # 将交叉熵向量求和，即可得到交叉熵
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.LEARNING_RATE).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),self.y_pred_cls)  # 由于input_y也是onehot编码，因此，调用tf.argmax(self.input_y)得到的是1所在的下表
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
