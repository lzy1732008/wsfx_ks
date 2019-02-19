# coding: utf-8
#包含5个输入关键参数：inputx,inputx_len,inputy,inputy_len,y
from __future__ import print_function

import os
import sys
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
from sklearn import metrics
import tensorflow.contrib.keras as kr

from wsfx2.code.models.matchpy1 import modelconfig,MatchPy
from wsfx2.code.train.loader import batch_iter5_test,data_load5,embedding_load
from wsfx2.code.train.evaluate import getwslist,wsevaluate

data_dir = '../../source/dataset/set_5'
trainpath = data_dir+'/train.txt'
validatepath = data_dir+'/val.txt'
testpath = data_dir +'/test.txt'
t_f = open(trainpath,'r',encoding='utf-8')
v_f = open(validatepath,'r',encoding='utf-8')
test_f = open(testpath,'r',encoding='utf-8')
ks_flag = 1

corpuspath = '../../source/wordvector/words.txt'
words_f = open(corpuspath,'r',encoding='utf-8')
embedding = embedding_load(words_f)

# save_dir  = '../../result/set5/matchpy1'  #修改处
# save_path = save_dir+'/checkpoints/30-50/best_validation'  # 最佳验证结果保存路径
# tensorboard_dir = save_dir+'/tensorboard/30-50/'  #修改处


save_dir  = '../../result/set5/matchpy1'  #修改处
ckpath = '30-50'
tbpath = '30-50'
save_path = save_dir+'/checkpoints/'+ckpath+'/best_validation'
tensorboard_dir = save_dir + '/tensorboard/' + tbpath


if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)


config = modelconfig(embedding)
model = MatchPy(config)




def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x1_len,x1_batch,x2_len,x2_batch,y_batch,  keep_prob):
    feed_dict = {
        model.X1_len: x1_len,
        model.X1:x1_batch,
        model.X2_len: x2_len,
        model.X2: x2_batch,
        model.Y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x1_len, x1_,x2_len, x2_,y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x1_)
    batch_eval = batch_iter5_test(x1_len, x1_, x2_len, x2_,y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x1_len_batch, x1_batch,x2_len_batch, x2_batch, y_batch in batch_eval:
        batch_len = len(x1_batch)
        feed_dict = feed_data(x1_len_batch, x1_batch,x2_len_batch, x2_batch, y_batch, 1.0)
        feed_dict[model.dpool_index] = model.dynamic_pooling_index(x1_len_batch, x2_len_batch,config.data1_maxlen,config.data2_maxlen)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    #结果可视化与存储
    tf.summary.scalar("loss", model.loss) #可视化loss
    tf.summary.scalar("accuracy", model.acc)  #可视化acc
    merged_summary = tf.summary.merge_all()   #将所有操作合并输出
    writer = tf.summary.FileWriter(tensorboard_dir) #将summary data写入磁盘

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x1_len, train_1, x2_len, train_2, train_output = data_load5(t_f,config)
    print('train len:',len(train_1))


    # print(train_3)
    val1_len, val_1, val2_len, val_2, val_output = data_load5(v_f,config)

    print('validation len:', len(val_1))

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter5(x1_len, train_1, x2_len, train_2,train_output, config.batch_size)
        for x1_len_batch, x1_batch, x2_len_batch, x2_batch, y_batch in batch_train:
            feed_dict = feed_data(x1_len_batch, x1_batch, x2_len_batch, x2_batch, y_batch,  config.dropout_keep_prob)
            dpool_batch = model.dynamic_pooling_index(x1_len_batch, x2_len_batch, config.data1_maxlen, config.data2_maxlen)
            feed_dict[model.dpool_index] = dpool_batch

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能

                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, val1_len, val_1, val2_len, val_2, val_output)  # 验证当前会话中的模型的loss和acc


                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break

def test():
    print("Loading test data...")
    start_time = time.time()
    x1_len_test, x1_test, x2_len_test, x2_test, y_test = data_load5(test_f, config)


    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x1_len_test, x1_test,x2_len_test, x2_test,  y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x1_test)
    num_batch = int((data_len) / batch_size)

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x1_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        dpool_index = model.dynamic_pooling_index(x1_len_test[start_id:end_id],x2_len_test[start_id:end_id],config.data1_maxlen, config.data2_maxlen)
        feed_dict = {
            model.X1_len: x1_len_test[start_id:end_id],
            model.X1: x1_test[start_id:end_id],
            model.X2_len: x2_len_test[start_id:end_id],
            model.X2: x2_test[start_id:end_id],
            model.dpool_index:  dpool_index,
            model.keep_prob: 1.0   #这个表示测试时不使用dropout对神经元过滤
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)   #将所有批次的预测结果都存放在y_pred_cls中



    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls,digits=4))#直接计算准确率，召回率和f值

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return y_test_cls,y_pred_cls

# train()
y_test_cls,y_pred_cls = test()
wsnamels = getwslist(model=model)
wsevaluate(y_test_cls, y_pred_cls,wsnamels)