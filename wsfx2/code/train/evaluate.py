# coding: utf-8

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
from sklearn import metrics
import tensorflow.contrib.keras as kr
import matplotlib.pyplot as plt

from wsfx2.code.models.cnn_model import TCNNConfig,TextCNN
from wsfx2.code.train.loader import batch_iter_test,data_load

data_dir = '../../source/dataset/set_2'
trainpath = data_dir+'/train.txt'
validatepath = data_dir+'/val.txt'
testpath = data_dir +'/test.txt'
testpath_fc = '../../source/dataset/set_1/test-分词.txt'
t_f = open(trainpath,'r',encoding='utf-8')
v_f = open(validatepath,'r',encoding='utf-8')
test_f = open(testpath,'r',encoding='utf-8')
test_fc_f = open(testpath,'r',encoding='utf-8')


ks_flag = 3 #kw level
n_number = 1 #n-gram
gate_n = 3
reg = False #defalut is false
times =1
ks_order = '123'
mirrorgate=  1
precessF = '2'
precessL = '1'
topK = 5 #defalut is 5
lastksinfo = 'Fasle' #defalut is true
singleuse = '3'
relu = 'False' #defalut is true

save_dir  = '../../result/set1/cnn_model'  #修改处
# save_path = save_dir+'/checkpoints/precessF:'+str(precessF)+'-MirrorGate:'+str(mirrorgate)+ks_order+'-time:'+str(times)+'noaddks-30-30-'+str(n_number)+'gram-gate'+str(gate_n)+'-'+str(reg)+'/best_validation'  # 最佳验证结果保存路径
# tensorboard_dir = save_dir+'/tensorboard/precessF:'+str(precessF)+'-MirrorGate:'+str(mirrorgate)+ks_order+'-time:'+str(times)+'noaddks-30-30-'+str(n_number)+'gram-gate'+str(gate_n)+'-'+str(reg)  #修改处
# ckpath = 'precessF:2MirrorGate:1231-time:3noaddks-30-30-1gram-gate1-False'
# tbpath = 'precessF:2MirrorGate:1231-time:3noaddks-30-30-1gram-gate1-False'

ckpath = '30-50'
tbpath = '30-50'

save_path = save_dir+'/checkpoints/'+ckpath+'/best_validation'
tensorboard_dir = save_dir + '/tensorboard/' + tbpath

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)


# config = TCNNConfig()
# model = TextCNN(config)




def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x1_batch,x2_batch,ks_batch,y_batch, keep_prob):
    feed_dict = {
        model.input_x1: x1_batch,
        model.input_x2: x2_batch,
        model.input_ks: ks_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x1_,x2_,ks_,y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x1_)
    batch_eval = batch_iter_test(x1_, x2_,ks_,y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x1_batch,x2_batch, ks_batch, y_batch in batch_eval:
        batch_len = len(x1_batch)
        feed_dict = feed_data(x1_batch,x2_batch,ks_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def wsevaluate(y_pred_cls,y_test_cls,wslist):
    print('y_pred_cls.len:',len(y_pred_cls))
    print('y_test_cls.len',len(y_test_cls))
    print('wslist.len:',len(wslist))
    pred_true = {}
    positive = {}
    pred_pos = {}
    for i in range(len(y_test_cls)):
        if pred_pos.get(wslist[i].strip()) == None:
            pred_pos[wslist[i].strip()] = 0
        if pred_true.get(wslist[i].strip()) == None:
            pred_true[wslist[i].strip()] = 0
        if positive.get(wslist[i].strip()) == None:
            positive[wslist[i].strip()] = 0

        if y_pred_cls[i] == 1:
            pred_pos[wslist[i]] += 1
        if y_test_cls[i] == 1:
            positive[wslist[i]] += 1
        if y_test_cls[i] == y_pred_cls[i] and y_pred_cls[i] == 1:
            pred_true[wslist[i]] += 1

    F1_ls = []
    wslist = list(set(wslist))
    for wsname in wslist:
        # print(pred_pos[wsname.strip()],positive[wsname.strip()],pred_true[wsname.strip()])
        if positive[wsname.strip()] == 0:
            print('Failed')
            continue
        else:
            recall = pred_true[wsname.strip()] / (positive[wsname.strip()])
            if pred_pos[wsname.strip()] == 0:
               preciosn = 0
            else:
               precision = pred_true[wsname.strip()]/(pred_pos[wsname.strip()])
            if recall + precision == 0:
                F1 = 0
            else:
                F1 = (2*recall*precision)/(precision+recall)
            F1_ls.append(F1)
            # print('F1:',F1)
    print('F1:',np.mean(np.array(F1_ls)))


def test():
    print("Loading test data...")
    start_time = time.time()
    x1_test, x2_test, ks_test, y_test = data_load(test_f, config, flag=ks_flag)
    print('x1_test len:',len(x1_test))



    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x1_test, x2_test, ks_test, y_test)
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
        feed_dict = {
            model.input_x1: x1_test[start_id:end_id],
            model.input_x2: x2_test[start_id:end_id],
            model.input_ks: ks_test[start_id:end_id],
            model.keep_prob: 1.0  # 这个表示测试时不使用dropout对神经元过滤
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls,feed_dict=feed_dict)  # 将所有批次的预测结果都存放在y_pred_cls中

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, digits=4))  # 直接计算准确率，召回率和f值

    # 混淆矩阵
    print("Confusion Matrix...")
    # cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    # print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return y_test_cls, y_pred_cls

def getwslist(model):
    namels = []
    lines = test_fc_f.read().split('\n')
    # print(lines)
    print(model.config.batch_size)
    max_idx = int(len(lines) / model.config.batch_size) * model.config.batch_size
    print('max_idx:',max_idx)
    for i in range(len(lines)):
        line = lines[i]
        if line.strip() == "":
            continue
        array = line.split('|')

        if len(array) < 5:
            continue

        namels.append(array[0])
    return namels


# y_test_cls, y_pred_cls= test()
# wsnamels = getwslist()
# wsevaluate(y_test_cls, y_pred_cls,wsnamels)


