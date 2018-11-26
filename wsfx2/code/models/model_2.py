#基于CNN，只使用最精细的那个先验知识，conv1d

import tensorflow as tf


class modelConfig(object):
    def __init__(self):
        self.EMBDDING_DIM = 128
        self.FACT_LEN = 30
        self.LAW_LEN = 30
        self.KS_LEN= 1

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
        self.input_x1 = tf.placeholder(tf.float32, [self.config.batch_size, self.config.FACT_LEN, self.config.EMBDDING_DIM],
                                        name='input_x1')
        self.input_x2 = tf.placeholder(tf.float32, [self.config.batch_size, self.config.LAW_LEN, self.config.EMBDDING_DIM],
                                        name='input_x2')
        self.input_ks = tf.placeholder(tf.float32, [self.config.batch_size, self.config.EMBDDING_DIM],name="input_ks")
        self.input_y = tf.placeholder(tf.int32, [self.config.batch_size, self.config.NUM_CLASS], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()
        return

    def cnn(self):
        new_x1 = self.gate(self.input_ks,self.input_x1)
        op1,op2 = self.conv(new_x1,self.input_x2)
        self.match(op1,op2)



    def gate(self, ks, inputx):
        with tf.name_scope("gate"):

            new_ks = tf.keras.backend.repeat(ks,self.config.FACT_LEN)



            weight_1 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM,self.config.EMBDDING_DIM],
                                                    stddev=0, seed=1),trainable=True, name='w1')
            weight_2 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, self.config.EMBDDING_DIM],
                                                    stddev=0, seed=1),trainable=True, name='w2')


            new_inputx = tf.reshape(inputx, shape=[self.config.EMBDDING_DIM,self.config.FACT_LEN * self.config.batch_size])
            new_ks2 = tf.reshape(new_ks, shape=[self.config.EMBDDING_DIM,self.config.FACT_LEN * self.config.batch_size])

            kw = tf.sigmoid(tf.reshape(tf.matmul(weight_1, new_inputx) + tf.matmul(weight_2,new_ks2),shape=inputx.shape))

            one_array = tf.ones(shape=kw.shape)

            new_vector = (one_array - kw) * inputx + kw * new_ks
        return new_vector


    def conv(self,inputx,inputy):
        with tf.name_scope("conv"):
            conv1 = tf.layers.conv1d(inputx,filters=self.config.FILTERS,kernel_size=self.config.KERNEL_SIZE,name='conv1')
            op1 = tf.reduce_max(conv1, reduction_indices=[1], name='gmp1')

            conv2 = tf.layers.conv1d(inputy,filters=self.config.FILTERS,kernel_size=self.config.KERNEL_SIZE,name='conv2')
            op2 = tf.reduce_max(conv2, reduction_indices=[1], name='gmp2')

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
            self.loss = tf.reduce_mean(cross_entropy)  # 将交叉熵向量求和，即可得到交叉熵
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.LEARNING_RATE).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),self.y_pred_cls)  # 由于input_y也是onehot编码，因此，调用tf.argmax(self.input_y)得到的是1所在的下表
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
