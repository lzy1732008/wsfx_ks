from wsfx2.code.util.excel_op import getrowls,getcolls,getexceldata
from wsfx2.code.util.file_fun import getlines
from wsfx2.code.util.str_op import getStrSegment
from wsfx2.code.pre_op.word2vec import vector,load_models

import os,random
import jieba.posseg as pos
'''
将数据集里面的数据对读取到txt，建立[训练集+验证集] 存储格式为:文书名|事实|法条|label\n,并且删除掉一半的负例
输入:exceldict:数据集目录,以及该txt文件存储位置
输出：无
'''
def fun1(exceldict,target):
    f = open(target, 'w', encoding='utf-8')
    dir = os.listdir(exceldict)

    for i in range(len(dir)):
        ex = dir[i]
        expath = exceldict + '/' + ex
        rows = getrowls(expath)  # 事实ls
        cols = getcolls(expath)  # 法条ls
        if len(rows) > 0 and len(cols) > 0:
            s = ''
            data = getexceldata(expath)
            # 统计0的个数和下标
            array_index = []
            count = 0
            for i in range(len(rows)):
                for j in range(len(cols)):
                    if str(rows[i]).strip() != '' and str(cols[j]).strip() != '':
                        if int(data[j][i]) == 0:
                            array_index.append(count)
                        count += 1


            # 随机生成被过滤的下标
            base = random.sample([i for i in range(len(array_index))], int(len(array_index) * 0.50))
            filer_index = []
            for index in base:
                filer_index.append(array_index[index])


            filer_index = []
            # 过滤下标是filter_index的负例样本


            count = 0
            for i in range(len(rows)):
                for j in range(len(cols)):
                    if rows[i].strip() == '' or cols[j].strip() == '':
                        pass
                    else:
                        if count not in filer_index:
                           s += ex + '|' + rows[i] + '|' + cols[j] + '|' + str(int(data[j][i])) + '\n'
                        # else:
                        #     print('filter')
                        count += 1

            f.write(s)
    f.close()

'''
训练数据
'''
# exceldict = '/Users/wenny/PycharmProjects/wsanalyse/wsfx/data/事实法条数据/训练集'
# target = '../../source/dataset/train-原始训练集.txt'
# fun(exceldict,target)

'''
测试数据
'''
# exceldict = '/Users/wenny/PycharmProjects/wsanalyse/wsfx/data/事实法条数据/测试集'
# target = '../../source/dataset/test-原始训练集.txt'
# fun1(exceldict,target)


#================================================================================================================

'''
将train-原始数据集.txt里面的每个法条对应的先验知识加入:
格式为:文书名|事实|法条|label|先验知识\n
输入:train-原始数据集,txt
'''
def fun2(ft_zs_f,data_f,newdata_f):
    #首先读取每个法条对应的先验知识存放在dict中
    ftzs_dict = {}
    lines = ft_zs_f.read().split('\n')
    for line in lines:
        zs = line.split('|')[0]
        ftname = line.split('|')[1]
        ftzs_dict[ftname] = zs
    lines = data_f.read().split('\n')
    s = ''
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        ft = line.split('|')[2]
        ftname = ft[:ft.index(":")]
        try:
           ftzs = ftzs_dict[ftname]
           s += line +'|' + ftzs + '\n'
        except:
            print(ftname)
    newdata_f.write(s)
    newdata_f.close()

'''
训练数据
'''
# ftzs_path = '../../source/法条/ft_nr_先验知识.txt'
# data_path = '../../source/dataset/train-原始训练集.txt'
# newdata_path = '../../source/dataset/train-添加先验知识.txt'
# f1 = open(ftzs_path,'r',encoding='utf-8')
# f2 = open(data_path,'r',encoding='utf-8')
# f3 = open(newdata_path,'w',encoding='utf-8')
# fun2(f1,f2,f3)


'''
测试数据
'''
# ftzs_path = '../../source/法条/ft_nr_先验知识.txt'
# data_path = '../../source/dataset/test-原始训练集.txt'
# newdata_path = '../../source/dataset/test-添加先验知识.txt'
# f1 = open(ftzs_path,'r',encoding='utf-8')
# f2 = open(data_path,'r',encoding='utf-8')
# f3 = open(newdata_path,'w',encoding='utf-8')
# fun2(f1,f2,f3)



'''
================================================================================================================
将train-添加先验知识.txt用分词，其中不同的词用空格分开，对于先验知识不同级别的用@分开
格式:文书名|事实|法条正文|label|先验知识\n
其中先验知识不同级别之间用的是@做分隔符
'''
def fun3(data_f,target_f):
    lines = data_f.read().split('\n')
    stoplist = getlines('../../source/stopwords.txt')
    cx_save = ['n','v','a','x']
    s = ''
    for line in lines:
        array = line.split('|')
        if len(array) != 5:
            print(line)
            break
        ss,ft,zs = array[1],array[2],array[4]
        ftname = ft[:ft.index(":")]
        ftzw = ft[ft.index(":"):]

        ssls = getStrSegment(ss,cx_save,stoplist)
        ftzwls = getStrSegment(ftzw,cx_save,stoplist)
        #当zs是"?"的情况
        zsstr = 'ft:' + ' '.join(getStrSegment(ftname, cx_save, stoplist)) +'@'
        if zs.strip() == "?": pass
        else:
              zs = str(zs).split(' ')
              for p in zs:
                  p_array = p.split(':')
                  if len(p_array) != 2:
                   print(line)
                   break
                  zslv = p.split(':')[0]
                  zsnr = p.split(':')[1]
                  zsstr += zslv + ':' + ' '.join(getStrSegment(zsnr, cx_save, stoplist)) + '@'
              s += array[0] + '|' +' '.join(ssls) + '|' + ' '.join(ftzwls) +'|'+ array[3] +'|' + zsstr +'\n'

    target_f.write(s)
    target_f.close()

'''
训练数据
'''
# data_f= open('../../source/dataset/train-添加先验知识.txt','r',encoding='utf-8')
# tartget_f = open('../../source/dataset/train-分词.txt','w',encoding='utf-8')
# fun3(data_f,tartget_f)

'''
测试数据
'''
# data_f= open('../../source/dataset/test-添加先验知识.txt','r',encoding='utf-8')
# tartget_f = open('../../source/dataset/test-分词.txt','w',encoding='utf-8')
# fun3(data_f,tartget_f)


'''
========================================================================================================================
向量化存储
输入:data_f:已经分好词的数据集
输出:target_f:向量化存储的数据集
格式:整体的分隔符仍然和原来一样，一个词向量内部用//表示
'''
def fun4(data_f, targte_f, model_ss, model_ft):
    news = ''
    lines = data_f.read().split('\n')
    for line in lines:
        array = line.split('|')
        if len(array) != 5:
            print(line)
            break
        news += array[0]+'|'

        ssls = array[1].split(' ')
        ftzwls = array[2].split(' ')
        zsls = array[4].split('@')
        for ss in ssls: news += '//'.join(map(str,list(vector(ss, model_ss)))) + ' '
        news += '|'
        for zw in ftzwls: news += '//'.join(map(str,list(vector(zw, model_ft)))) + ' '
        news += '|'+array[3] +'|'
        news += zsVector(zsls,model_ft,flag=1)+'\n'
    targte_f.write(news)


def zsVector(zwls, word_m,flag=1):
    s =''
    #只对最细致的那个level进行向量化
    if flag==1:
        obls = zwls[-2].split(':')[1].split(' ')
        for ob in obls:
            s += '//'.join(map(str,list(vector(ob,word_m)))) +' '
    return s

'''
训练数据
'''
# data_f= open('../../source/dataset/set_1/train-分词.txt','r',encoding='utf-8')
# tartget_f = open('../../source/dataset/set_2/train-向量化.txt','w',encoding='utf-8')
# model_ss = load_models('../../source/wordvector/ssmodel_size128.model')
# model_ft = load_models('../../source/wordvector/ssmodel_size128.model')
# fun4(data_f,tartget_f,model_ss,model_ft)

'''
测试数据
# '''
# data_f= open('../../source/dataset/set_1/test-分词.txt','r',encoding='utf-8')
# tartget_f = open('../../source/dataset/set_2/test-向量化.txt','w',encoding='utf-8')
# model_ss = load_models('../../source/wordvector/ssmodel_size128.model')
# model_ft = load_models('../../source/wordvector/ssmodel_size128.model')
# fun4(data_f,tartget_f,model_ss,model_ft)

# ========================================================================================================================
'''
切分训练集数据:训练集/验证集 = 9000/2447
输入:向量化好的数据集txt
输出:划分好的数据集(为了能够计算基于文书的效率，我们随机选取几篇文书作为验证集)
'''
def fun4(data_f,wsls,targetpath):
    #从wsls目录中选取作为验证集的文书名字
    val_index = random.sample([i for i in range(len(wsls))],50)
    print(val_index)
    val_names = []
    for index in val_index: val_names.append(wsls[index])

    val_str = ''
    train_str = ''
    lines = data_f.read().split('\n')
    for line in lines:
        name = line.split('|')[0].strip()
        if name in val_names:
            val_str += line + '\n'
        else:
            train_str += line + '\n'
    f_v = open(os.path.join(targetpath,'val.txt'),'w',encoding='utf-8')
    f_t = open(os.path.join(targetpath,'train.txt'),'w',encoding='utf-8')
    f_v.write(val_str)
    f_v.close()
    f_t.write(train_str)
    f_t.close()


# exceldict = '/Users/wenny/PycharmProjects/wsanalyse/wsfx/data/事实法条数据/训练集'
# wsls = os.listdir(exceldict)
# data_f = open('../../source/dataset/set_2/train-向量化.txt','r',encoding='utf-8')
# fun4(data_f,wsls,'../../source/dataset/set_2')
























