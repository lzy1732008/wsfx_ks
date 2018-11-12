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
import math
from wsfx2.code.util.excel_op import createx

from wsfx2.code.models.model_8 import modelConfig,CNN
from wsfx2.code.train.loader import batch_iter_test,data_load

data_dir = '../../source/dataset/test'
testpath = data_dir +'/test.txt'
testpath_fc = data_dir+'test-fc.txt'
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

save_dir  = '../../result/set4/model8'  #修改处
ckpath = 'precessF:2MirrorGate:1231-time:3noaddks-30-30-1gram-gate1-False'
tbpath = 'precessF:2MirrorGate:1231-time:3noaddks-30-30-1gram-gate1-False'
# ckpath='norelu3'
# tbpath = 'norelu3'

save_path = save_dir+'/checkpoints/'+ckpath+'/best_validation'
tensorboard_dir = save_dir + '/tensorboard/' + tbpath

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)


config = modelConfig()
model = CNN(config)




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





def drawFigure(data):
    plt.imshow(data, interpolation='nearest', cmap='bone', origin='lower')
    plt.colorbar()
    plt.xticks()
    plt.yticks()
    plt.show()

def disOS(v1,v2):
    dist = np.linalg.norm(v1-v2)
    return 1.0/(1.0 + dist)

def discosin(v1,v2):
    num = np.sum(v1*v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos = num/denom
    sim = 0.5 + 0.5 * cos
    return sim

def countdis(x1,ks):
    disls = []
    for k in ks:
        disls.append(disOS(x1,k))
    return disls



def dismatrix(x1ls,ks):
    M = []
    for x1 in x1ls:
        M.append(discosin(x1,ks))
    return M

def cmpdis(x1_test,new_x1,ks_test):
    M1 = dismatrix(x1_test,ks_test)
    M2 = dismatrix(new_x1,ks_test)
    return M1,M2


def test():
    print("Loading test data...")
    x1_test, x2_test, ks_test, y_test = data_load(test_f, config, flag=ks_flag)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    feed_dict ={
        model.input_x1: x1_test,
        model.input_x2: x2_test,
        model.input_ks: ks_test,
        model.keep_prob: 1.0  # 这个表示测试时不使用dropout对神经元过滤
    }
    y_pred_cls, new_x1, new_x1_mean, new_x2 = session.run([model.y_pred_cls, model.new_x1, model.new_x1_mean, model.new_x2],
                                                  feed_dict=feed_dict)

    return y_pred_cls ,x1_test,ks_test, new_x1, new_x1_mean, new_x2



y_pred_cls, x1_test,ks_test, new_x1, new_x1_mean, new_x2 = test()
M1,M2 = cmpdis(x1_test[1,-5:],new_x1[1,-5:],ks_test[1])
# drawFigure(data2)
createx('123_1456165.xml_ft2jl.xls1', rows=['1','2','3'], colums=['乙', '胸部' ,'损伤' ,'程度' ,'轻伤'], data=M1 ,dir='../../record')





