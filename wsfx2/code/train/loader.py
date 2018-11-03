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
    n_vector = [list(map(float, list(filter(lambda x: x.strip() != '', ss.split('//'))))) for ss in ssls]
    return n_vector

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


    train_1 = kr.preprocessing.sequence.pad_sequences(np.array(input_x1), config.FACT_LEN-1)
    train_2 = kr.preprocessing.sequence.pad_sequences(np.array(input_x2), config.LAW_LEN-1)
    train_ks = np.array(input_ks)

    return train_1,train_2,train_ks,np.array(input_y)

#给每个文本加上一个start向量
def addStart(inputx,inputy):
    start = np.random.randn(128)
    inputx = list(inputx)
    inputy = list(inputy)
    new_inputx = []
    new_inputy = []

    for s in inputx:
        s.insert(0,start)
        new_inputx.append(s)

    for s in inputy:
        s.insert(0,start)
        new_inputy.append(s)
    return new_inputx, new_inputy

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
        ft_len = int(array[3])
        ss_len = int(array[1])
        if len(array) < 6 or ft_len*ss_len == 0:
            print(line)
            continue

        ssls = list(map(int,list(filter(lambda x:x.strip()!='',array[2].split(' ')))))
        ftzwls = list(map(int,list(filter(lambda x:x.strip()!='',array[4].split(' ')))))
        label = int(array[5].strip())
        input_x1.append(ssls)
        input_x2.append(ftzwls)
        if ss_len > config.data1_maxlen: ss_len = config.data1_maxlen
        if ft_len > config.data2_maxlen: ft_len = config.data2_maxlen
        x1_len.append(ss_len)
        x2_len.append(ft_len)
        if label == 0:
            input_y.append([1, 0])
        else:
            input_y.append([0, 1])

    train_1 = kr.preprocessing.sequence.pad_sequences(np.array(input_x1), config.data1_maxlen)
    train_2 = kr.preprocessing.sequence.pad_sequences(np.array(input_x2), config.data2_maxlen)

    return np.array(x1_len), np.array(train_1),  np.array(x2_len), np.array(train_2),  np.array(input_y)

#used with data_load5
def batch_iter5(x1_len, x1, x2_len, x2,  y,  batch_size=128):
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
        yield x1_len_shuffle[start_id:end_id],x1_shuffle[start_id:end_id],x2_len_shuffle[start_id:end_id], x2_shuffle[start_id:end_id],  y_shuffle[start_id:end_id]


def embedding_load(words_f):
    cpslist = words_f.read().split('\n')
    embedding_dict = {}
    for i in range(len(cpslist)): embedding_dict[cpslist[i].strip()] = i + 1
    embedding_dict['NOT_FOUND'] = len(cpslist) + 1
    embedding_dict['PAD'] = 0
    embedding = np.float32(np.random.uniform(-0.02, 0.02, [len(embedding_dict), 50]))
    return embedding


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
