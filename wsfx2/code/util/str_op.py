import re
import jieba.posseg as pos

def cutcontent(content):
    return list(filter(lambda x: x.strip() != '', re.split('；|。', content)))

def getStrSegment(content,cx_save,stoplist):
    words = pos.cut(content)
    sls = []
    for word, cx in words:
        if cx in cx_save:
            if word in stoplist:
                pass
            else:
                sls.append(word)
    return sls

def showClassNum(content):
    lines = content.split('\n')
    class_0, class_1 = 0, 0

    for line in lines:
        if len(line.split('|')) <5:
            continue
        label = line.split('|')[3][0]
        if label == '0':
            class_0 += 1
        else:
            class_1 += 1
    print('0:',class_0)
    print('1:',class_1)

# f = open('../../source/dataset/set_1/val-分词.txt','r',encoding='utf-8')
# lines = f.read()
# showClassNum(lines)

