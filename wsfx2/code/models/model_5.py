#添加多级先验知识，并且上一层级得到的[1，d]的score会传入下一层的下一级的计算中，使用的每层计算的权重是两个[d,1]的


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

    '''
 这个门机制如下：采用的是上一级得到的相似概率si会传递到下一级，然后再对s和ks以及inputx输入MLP中计算得出分配在每个的概率，然后用这个概率计算new_vector
 s1 = x * w * k_1
 s2 = x * w * [k_2;s1]
 s3 = x * w * [k_3;s2]
 M = [k_1;s1,k_2;s2,k_3;s3,inputx;(4-s1-s2-s3-s4)] shape:[4,2d]
 pw = M*W  shape:[None,l,4,d]
 new_vector = pw[0]*k_1 + pw[1]*k_2 + pw[2]* k_3 + pw[4] * inputx
    '''
    def gate(self, ks, inputx):
        with tf.name_scope("gate"):
            weight_1 = tf.Variable(tf.random_normal([self.config.EMBDDING_DIM, self.config.FACT_LEN],
                                                    stddev=0, seed=1),trainable=True, name='w1')
            weight_2 = tf.Variable(tf.random_normal([2 * self.config.EMBDDING_DIM, self.config.FACT_LEN],
                                                    stddev=0, seed=2),trainable=True, name='w2')
            weight_3 = tf.Variable(tf.random_normal([2 * self.config.EMBDDING_DIM, self.config.FACT_LEN],
                                                   stddev=0, seed=3), trainable=True, name='w3')
            weight_4 = tf.Variable(tf.random_normal([2 * self.config.EMBDDING_DIM, self.config.EMBDDING_DIM],
                                                   stddev=0, seed=4), trainable=True, name='w4')


            k_1_init, k_2_init, k_3_init = ks[:,0,:], ks[:,1,:], ks[:,2,:] #[None,d]
            k_1 = tf.reshape(tf.keras.backend.repeat_elements(k_1_init,rep=self.config.FACT_LEN,axis=1),
                                shape=[-1,self.config.FACT_LEN,self.config.EMBDDING_DIM])
            k_2 = tf.reshape(tf.keras.backend.repeat_elements(k_2_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, self.config.EMBDDING_DIM])
            k_3 = tf.reshape(tf.keras.backend.repeat_elements(k_3_init, rep=self.config.FACT_LEN, axis=1),
                             shape=[-1, self.config.FACT_LEN, self.config.EMBDDING_DIM])
            print('inputx.shape:',inputx.shape)
            fun1 = tf.einsum('abc,cd->abd', k_1, weight_1)
            ksw_1 = tf.sigmoid(tf.einsum('abd,adf->abf',fun1,inputx)) #[batch,l,d]

            fun2 = tf.einsum('abc,cd->abd', tf.concat([k_2,ksw_1],axis=2), weight_2)
            ksw_2 = tf.sigmoid(tf.einsum('abd,adf->abf',fun2,inputx))#[batch,l,d]

            fun3 = tf.einsum('abc,cd->abd', tf.concat([k_3,ksw_2],axis=2), weight_3)
            ksw_3 = tf.sigmoid(tf.einsum('abd,adf->abf',fun3,inputx)) #[batch,l,d]


            ksw_concat = tf.concat([ksw_1,ksw_2,ksw_3],axis=2) #[None,l,3*d]
            ksw_concat = tf.reshape(ksw_concat,shape=[-1,self.config.FACT_LEN,3,self.config.EMBDDING_DIM])#[None,l,3,d]
            ksw_sum = tf.reduce_sum(ksw_concat,axis=2) #[None,l,1,d]
            one = 4 * tf.ones(shape=tf.shape(ksw_sum)) #[None,l,1,d]
            p_x = one-ksw_sum #[None,l,1,d]
            p_x = tf.expand_dims(p_x,axis=2)
            inputx_epd = tf.expand_dims(inputx,axis=2)


            ks_epd = tf.reshape(tf.keras.backend.repeat_elements(ks,rep=self.config.FACT_LEN,axis=1),
                                shape=[-1,self.config.FACT_LEN,self.config.KS_LEN, self.config.EMBDDING_DIM]) #[None,l,3,d]
            ks_p_mul = tf.concat([ks_epd,ksw_concat],axis=3)#[None,l,3,2d]
            M = tf.concat([ks_p_mul,tf.concat([inputx_epd,p_x],axis=3)],axis=2) #[None,l,4,2d]
            pw = tf.sigmoid(tf.einsum('abcd,df->abcf',M,weight_4)) #[None,l,4,d]

            kw_input = tf.concat([ks_epd,inputx_epd],axis=2) #[None,l,4,d]
            temp1 = tf.reduce_sum(pw * kw_input,axis=2)#[None,l,1,d]
            new_vector = tf.reshape(temp1,shape=[-1,self.config.FACT_LEN,self.config.EMBDDING_DIM])
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
