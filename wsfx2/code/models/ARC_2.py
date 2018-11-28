import tensorflow as tf

class modelConfig(object):
    def __init__(self):
        self.EMBDDING_DIM = 128
        self.seq_length_1 = 30
        self.seq_length_2 = 30

        self.FILTERS = 30
        self.KERNEL_SIZE = 3  # 卷积核尺寸
        self.LAYER_NUM  = 2
        self.filters_2d = [256, 128]
        self.kernel_size_2d = [[3, 3], [3, 3]]
        self.mpool_size_2d = [[2, 2], [2, 2]]


        self.LAYER_UNITS = 100
        self.NUM_CLASS = 2

        self.LEARNING_RATE = 0.001
        self.batch_size = 128
        self.num_epochs = 200
        self.save_per_batch = 10
        self.print_per_batch = 10
        self.dropout_keep_prob = 0.5


class ARC2model(object):
    def __init__(self, config):
        self.config = config
        self.input_x_1 = tf.placeholder(tf.float32, [None, self.config.seq_length_1 , self.config.EMBDDING_DIM],
                                       name='input_x1')
        self.input_x_2 = tf.placeholder(tf.float32, [None, self.config.seq_length_2 , self.config.EMBDDING_DIM],
                                       name='input_x2')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.NUM_CLASS],
                                      name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()
        return

    def cnn(self):
        concat = tf.concat([self.input_x_1,self.input_x_2],axis=-1)
        with tf.name_scope("layer1"):
            layer1_input=tf.layers.conv1d(concat, filters=self.config.FILTERS, kernel_size=self.config.KERNEL_SIZE,
                             name='conv1',activation=tf.nn.relu,padding='SAME')
            # layer1_reshaped = tf.reshape(layer1_input,shape=[self.config.FACT_LEN,self.config.LAW_LEN,-1])
            layer1_expd = tf.expand_dims(layer1_input,axis=3)
            layer1_output = tf.layers.max_pooling2d(inputs=layer1_expd,pool_size=(2,2),strides=(2,2))

        with tf.name_scope("layer2"):
            for i in range(self.config.LAYER_NUM):
                z = tf.layers.conv2d(inputs=layer1_output,filters=self.config.filters_2d[i],
                                     kernel_size=self.config.kernel_size_2d[i],padding='SAME',activation=tf.nn.relu)
                z = tf.layers.max_pooling2d(inputs=z,pool_size=self.config.mpool_size_2d[i],strides=self.config.mpool_size_2d[i])

        flatten = tf.layers.Flatten()(z)
        flatten = tf.contrib.layers.dropout(flatten, self.keep_prob)
        with tf.name_scope("MLP"):
            fc = tf.layers.dense(inputs=flatten, units=self.config.LAYER_UNITS, use_bias=True,
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










