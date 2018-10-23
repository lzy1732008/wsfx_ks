# coding: utf-8

import sys
import tensorflow.contrib.keras as kr
from collections import Counter
import pandas as pd
import numpy as np

import numpy as np


if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def data_convert(vectors):
    ssls = list(filter(lambda x:x.strip() != '', vectors))
    return [list(map(float, list(filter(lambda x: x.strip() != '', ss.split('//'))))) for ss in ssls]

#2 inputs
def data_load2(data_f,config):
    input_x1, input_x2,  input_y = [], [], []
    lines = data_f.read().split('\n')
    for i in range(len(lines)):
        line = lines[i]
        print('index:', i)
        if line.strip() == "":
            continue

        array = line.split('|')
        if len(array) < 5:
            continue
        ssls = array[1].split(' ')
        ftzwls = array[2].split(' ')
        label = int(array[3].strip())
        input_x1.append(data_convert(ssls))
        input_x2.append(data_convert(ftzwls))
        if label == 0:
            input_y.append([1, 0])
        else:
            input_y.append([0, 1])

    train_1 = kr.preprocessing.sequence.pad_sequences(np.array(input_x1), config.seq_length_1)
    train_2 = kr.preprocessing.sequence.pad_sequences(np.array(input_x2), config.seq_length_2)

    return train_1, train_2,  np.array(input_y)

#used with data_load2
def batch_iter2(x1, x2,  y, batch_size=128):
    """生成批次数据"""
    data_len = len(x1)
    num_batch = int(data_len / batch_size)

    indices = np.random.permutation(np.arange(data_len)) #洗牌
    x1_shuffle = x1[indices]
    x2_shuffle = x2[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x1_shuffle[start_id:end_id],x2_shuffle[start_id:end_id],  y_shuffle[start_id:end_id]


#3 inputs
def data_load(data_f, config, flag=3):
    input_x1,input_x2,input_ks,input_y = [], [], [], []
    lines = data_f.read().split('\n')
    for i in range(len(lines)):
        line = lines[i]
        print('index:',i)
        if line.strip() == "":
            continue

        array = line.split('|')
        if len(array) < 5:
            continue
        ssls = array[1].split(' ')
        ftzwls = array[2].split(' ')
        label = int(array[3].strip())
        zsls = array[4].split(' ')
        input_x1.append(data_convert(ssls))
        input_x2.append(data_convert(ftzwls))
        if label==0: input_y.append([1,0])
        else: input_y.append([0,1])
        if flag == 1:#mean vectors of words of 'jie'
           zs_matrix = np.mean(data_convert(zsls),axis=0)
        else:
            zs_matrix = data_convert(zsls)
        input_ks.append(zs_matrix)


    train_1 = kr.preprocessing.sequence.pad_sequences(np.array(input_x1), config.FACT_LEN)
    train_2 = kr.preprocessing.sequence.pad_sequences(np.array(input_x2), config.LAW_LEN)
    train_ks = np.array(input_ks)

    return train_1,train_2,train_ks,np.array(input_y)

#5-input
def data_load5(data_f,config):
    input_x1, x1_len, x2_len, input_x2,  input_y = [], [], [], [], []
    lines = data_f.read().split('\n')
    for i in range(len(lines)):
        line = lines[i]
        print('index:', i)
        if line.strip() == "":
            continue

        array = line.split('|')
        if len(array) < 6:
            print(line)
            continue
        ss_len = int(array[1])
        ssls = map(int,list(filter(lambda x:x.strip()!='',array[2].split(' '))))
        ft_len = int(array[3])
        ftzwls = map(int,list(filter(lambda x:x.strip()!='',array[4].split(' '))))
        label = int(array[5].strip())
        input_x1.append(ssls)
        input_x2.append(ftzwls)
        x1_len.append(ss_len)
        x2_len.append(ft_len)
        if label == 0:
            input_y.append([1, 0])
        else:
            input_y.append([0, 1])

    train_1 = kr.preprocessing.sequence.pad_sequences(np.array(input_x1), config.data1_maxlen)
    train_2 = kr.preprocessing.sequence.pad_sequences(np.array(input_x2), config.data2_maxlen)

    return x1_len, train_1,  x2_len, train_2,  np.array(input_y)

#used with data_load5
def batch_iter5(x1_len, x1, x2_len, x2,  y, batch_size=128):
    """生成批次数据"""
    data_len = len(x1)
    num_batch = int(data_len / batch_size)

    indices = np.random.permutation(np.arange(data_len)) #洗牌
    x1_len_shuffle = x1_len[indices]
    x1_shuffle = x1[indices]
    x2_len_shuffle = x2_len[indices]
    x2_shuffle = x2[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x1_len_shuffle[start_id:end_id],x1_shuffle[start_id:end_id],x2_len_shuffle[start_id:end_id],x2_shuffle[start_id:end_id],  y_shuffle[start_id:end_id]




#2-gram sum
def data_ngram(inputx,number):
    new_inputx = []
    for batch_a in inputx:
        batch_t = pd.DataFrame(batch_a)
        batch_sum = batch_a
        for _ in range(number-1):
            batch_sum = np.array(batch_t.shift() + batch_sum)
            batch_t = batch_t.shift()
        batch_sum = (np.array(batch_sum)).tolist()
        batch_sum.extend ([[0] * 128]*(number-1))
        batch_sum = np.array(batch_sum[number-1:])
        new_inputx.append(batch_sum)
    return np.array(new_inputx)


def batch_iter(x1, x2, ks, y, batch_size=128):
    """生成批次数据"""
    data_len = len(x1)
    num_batch = int(data_len / batch_size)

    indices = np.random.permutation(np.arange(data_len)) #洗牌
    x1_shuffle = x1[indices]
    x2_shuffle = x2[indices]
    ks_shuffle = ks[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x1_shuffle[start_id:end_id],x2_shuffle[start_id:end_id], ks_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
