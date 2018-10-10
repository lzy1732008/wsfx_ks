#输入法条的关键词，使用word2vec获取法条关键词的同义词，注意:输入文书作为语料库，用文书的QW内容
import os
import jieba.posseg as pos
import gensim
from gensim.models import word2vec
from wsfx2.code.util.file_fun import getlines


def buildmodel(wspath,corpuspath,modelpath,spwordpath):
    print('build model......')
    setCor(wspath,corpuspath,spwordpath)
    print('start.....')
    sentence = word2vec.LineSentence(corpuspath)
    model = word2vec.Word2Vec(sentence,min_count=5,size = 128)
    print('saveing.....')
    model.save(modelpath)
    print('built......')
    print('end....')


#将全文中指定词性的词删除
def filterwordwithcx(cutre,cxlist,spwordpath):
    wordlist = []
    print('filterword....')
    stopwords = getlines(spwordpath)
    for (w,k) in cutre:
        if k not in cxlist and w not in stopwords:
            wordlist.append(w)
    return wordlist

def setCor(dicpath,corpuspath,spwordpath):
    print('setCor:'+dicpath)
    filepathlist = os.listdir(dicpath)
    index = 0
    cxlist = ['x','p','nr','uj']
    with open(corpuspath,'w',encoding='UTF-8') as f:
        for filepath in filepathlist:
            print('index', index)
            index += 1
            # content = getQW(os.path.join(dicpath,filepath)).attrib['value']
            content = open(os.path.join(dicpath,filepath),'r',encoding='utf-8').read()
            contentcut = pos.cut(content)
            content_filter = filterwordwithcx(contentcut, cxlist, spwordpath)
            for word in content_filter:
                f.write(word+' ')
            f.write('\n')

def load_models(model_path):
    return gensim.models.Word2Vec.load(model_path)

def vector(v,model):
    try:
        return model[v]
    except:
        return [0]*128

if __name__=='__main__':
    ws_path='../../source/法律全文'
    corpus_path = '../../source/wordvector/corpus.txt'
    model_path = '../../source/wordvector/lawmodel_size128.model'
    stopwordspath = '../../source/stopwords.txt'
    # buildmodel(ws_path,corpus_path,model_path,stopwordspath)
    model = load_models(model_path)
    n = vector('刑期',model)
    print(n)
    # print('/'.join(list(n)))
    # ls = vector('AKG',model)
    # print(ls)
    # s = 0
    # for n in ls:
    #     s +=n
    # print(s/len(ls))
    #
    # ls = model['受伤']
    # s = 0
    # for n in ls:
    #     s += n
    # print(s / len(ls))


