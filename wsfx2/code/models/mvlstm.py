
import tensorflow as tf

class modelconfig(object):
    def __init__(self):
        self.EMBDDING_DIM = 128
        self.NUM_CLASS = 2
        self.seq_length_1 = 30
        self.seq_length_2 = 30

        self.OUTPUT_DIM = 16
        self.HIDDEN_DIM = 50
        self.K = 5


        self.LEARNING_RATE = 0.001
        self.batch_size = 128
        self.num_epochs = 200
        self.save_per_batch = 10
        self.print_per_batch = 10
        self.dropout_keep_prob = 0.5
        self.num_layers = 1

class MVLSTM(object):
    def __init__(self,config):
        self.config = config
        self.input_x_1 = tf.placeholder(tf.float32,
                                       [None, self.config.seq_length_1, self.config.EMBDDING_DIM],
                                       name='input_x1')
        self.input_x_2 = tf.placeholder(tf.float32,
                                       [None, self.config.seq_length_2, self.config.EMBDDING_DIM],
                                       name='input_x2')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.NUM_CLASS], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.mv_lstm()
        return

    def mv_lstm(self):
        #首先输入一个bilstm
        _outputs_1, state_1 = self.BiLSTM(inputx=self.input_x_1,index=1)
        _outputs_2, state_2 = self.BiLSTM(inputx=self.input_x_2, index=2)
        fw_state_1, bw_state_1 = _outputs_1[0], _outputs_1[1]

        embed_1 = tf.concat([fw_state_1,bw_state_1], axis=2) #[batch_size,maxtime,hidden_dim]

        fw_state_2, bw_state_2 = _outputs_2[0], _outputs_2[1]
        embed_2 = tf.concat([fw_state_2, bw_state_2], axis=2)

        #计算相似度
        simlarity = self.simlarity(embed_1,embed_2)

        #max-pooling
        max_result = self.max_pooling(simlarity)

        #MLP
        fc2 = self.MLP(max_result)
        self.y_pred_cls = tf.argmax(fc2,axis=1)

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2,
                                                                    labels=self.input_y)  # 对logits进行softmax操作后，做交叉墒，输出的是一个向量
            self.loss = tf.reduce_mean(cross_entropy)  # 将交叉熵向量求和，即可得到交叉熵
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.LEARNING_RATE).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),self.y_pred_cls)  # 由于input_y也是onehot编码，因此，调用tf.argmax(self.input_y)得到的是1所在的下表
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.config.HIDDEN_DIM, forget_bias= 1.0, state_is_tuple=True)

    def dropout(self):  # 为每一个rnn核后面加一个dropout层
        return tf.contrib.rnn.DropoutWrapper(self.lstm_cell(), output_keep_prob=self.keep_prob)

    def BiLSTM(self,inputx,index):
        with tf.variable_scope("bi-lstm"+str(index)):
            cell_fw = [self.dropout() for _ in range(self.config.num_layers)]
            cell_bw = [self.dropout() for _ in range(self.config.num_layers)]

            cell_fw = tf.contrib.rnn.MultiRNNCell(cell_fw, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.MultiRNNCell(cell_bw, state_is_tuple=True)
            # cell_fw, cell_bw = bi_lstm()

            init_state_fw = cell_fw.zero_state(batch_size=self.config.batch_size, dtype=tf.float32)
            init_state_bw = cell_bw.zero_state(batch_size=self.config.batch_size, dtype=tf.float32)

            _outputs_1, state_1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                  inputs=inputx,
                                                                  initial_state_fw=init_state_fw,
                                                                  initial_state_bw=init_state_bw)
            return _outputs_1,state_1

    def simlarity(self,embed_1,embed_2):
        with tf.name_scope('simlarity'):
            w1 = tf.get_variable('w1',
                                 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32),
                                 dtype=tf.float32,
                                 shape=[self.config.OUTPUT_DIM, 2 * self.config.HIDDEN_DIM, 2 * self.config.HIDDEN_DIM],trainable=True)
            v1 = tf.get_variable('v1',
                                 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32),
                                 dtype=tf.float32,
                                 shape=[4 * self.config.HIDDEN_DIM, self.config.seq_length_2, self.config.OUTPUT_DIM],trainable=True)

            forward_result1 = []
            for i in range(self.config.OUTPUT_DIM):
                r1 = tf.einsum('bld,df->blf',embed_1,w1[i])
                r2 = tf.einsum('bld,bdf->blf',r1,tf.transpose(embed_2,[0,2,1]))
                forward_result1.append(r2)

            forward_result2 = tf.einsum('bld,dfo->blfo',tf.concat([embed_1,embed_2],axis=2),v1)

            forward_result3 = tf.nn.relu(tf.reshape(forward_result1,shape=[-1,self.config.seq_length_1,self.config.seq_length_1,self.config.OUTPUT_DIM])+forward_result2)

            return forward_result3

    def max_pooling(self,simlarity):
        with tf.name_scope("max_pooling"):
            s2 = tf.reshape(simlarity, shape=[-1, self.config.seq_length_1 * self.config.seq_length_1])
            value,indexs = tf.nn.top_k(s2, k=self.config.K, sorted=False)
            max_result = tf.reshape(value, shape=[-1, self.config.OUTPUT_DIM * self.config.K])
            return max_result


    def MLP(self,max_result):
        with tf.name_scope("MLP"):
            w2 = tf.get_variable('w2',
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1,
                                                                                  dtype=tf.float32),
                                      dtype=tf.float32, shape=[self.config.K * self.config.OUTPUT_DIM, 64],trainable=True)
            b2 = tf.get_variable('b2', initializer=tf.constant_initializer(), dtype=tf.float32, shape=[64],trainable=True)
            fc1 = tf.nn.relu(tf.matmul(max_result, w2) + b2)
            fc1_drop = tf.nn.dropout(fc1, self.keep_prob)

            w3 = tf.get_variable('w3',
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1,
                                                                                  dtype=tf.float32),
                                      dtype=tf.float32, shape=[64, 2],trainable=True)
            b3 = tf.get_variable('b3', initializer=tf.constant_initializer(), dtype=tf.float32, shape=[2],trainable=True)
            fc2 = tf.nn.softmax(tf.matmul(fc1_drop, w3) + b3)
            return fc2
















