import lxml.etree
import os
def getQW(path):
    tree = lxml.etree.parse(path)
    root = tree.getroot()
    for qw in root:
        return qw

def getRDSS(path):
    content = ''
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == 'AJJBQK':
            for ajjbqkchild in qwchild:
                if ajjbqkchild.tag == 'BSSLD':
                    for bssldchild in ajjbqkchild:
                        if bssldchild.tag == 'ZJXX':
                            for zjxxchild in bssldchild:
                                if zjxxchild.tag == 'ZJFZ':
                                    for zjfzchild in zjxxchild:
                                        if zjfzchild.tag == 'RDSS':
                                            content = zjfzchild.attrib['value']
    return content
#指控段落
def getZKDL(path):
    content = ''
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == 'AJJBQK':
            for ajjbqkchild in qwchild:
                if ajjbqkchild.tag == 'ZKDL':
                    content = ajjbqkchild.attrib['value']
    return content

#从新填充了法条内容的文书里提取法条列表
def getFTList(path):
    ftnamelist = []
    ftnrlist = []
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == 'YYFLNR':
            for yyflfzchild in qwchild:
                if yyflfzchild.tag == 'FLNRFZ':
                    for flnrfzchild in yyflfzchild:
                        flag = 0
                        if flnrfzchild.tag == 'FLMC':
                            flmc = flnrfzchild.attrib['value']
                            flag += 1
                        if flnrfzchild.tag == 'FLNR':
                            flnr = flnrfzchild.attrib['value']
                            flag += 2
                        if flag == 2 and flmc and flnr and flnr != 'NOT FOUND':
                            ftnamelist.append(flmc)
                            ftnrlist.append(flnr)

    return ftnamelist,ftnrlist

#文书QW下面的节点内容获取,如文首、诉讼情况、案件基本情况、裁判分析过程、判决结果这几个的value

def getQWChildContent(path,childname):
    content = ''
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == childname:
            content += qwchild.attrib['value']

    return content



def getFTfromQW(path):
    ftls = []
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == 'CPFXGC':
            for cpfxgcchild in qwchild:
                if cpfxgcchild.tag == 'CUS_FLFT_FZ_RY':
                    for fz in cpfxgcchild:
                        if fz.tag == 'CUS_FLFT_RY':
                            ftls.append(fz.attrib['value'])
    return ftls




# 获取事实内容
def getSSMatchObject(wspath):
    return getRDSS(wspath) + getZKDL(wspath)


# 获取结论内容
def getJLMatchObject(wspath):
    return getQWChildContent(wspath, 'CPFXGC') + getQWChildContent(wspath, 'PJJG')

#获取交通肇事罪的证据记录列表
def getZJ(wspath):
    zjlist = []
    qw = getQW(wspath)
    for qwchild in qw:
        if qwchild.tag == 'AJJBQK':
            for ajjbqkchild in qwchild:
                if ajjbqkchild.tag == 'BSSLD':
                    for bssldchid in ajjbqkchild:
                        if bssldchid.tag == 'ZJXX':
                            for zjxxchild in bssldchid:
                                if zjxxchild.tag == 'ZJFZ':
                                    for zjfzchild in zjxxchild:
                                        if zjfzchild.tag == 'ZJJL':
                                            zjlist.append(zjfzchild.attrib['value'])
    return zjlist


#获取xml任意路径的value值
def getnodecontent(wspath,xmlpath):
    pathlist = xmlpath.split('/')
    print(pathlist)
    tree = lxml.etree.parse(wspath)
    root = tree.getroot()
    point = root
    index = 0
    while(index < len(pathlist)):
        for child in point:
            if child.tag == pathlist[index]:
                point = child
                index += 1
                break
    valuelist = []
    parent = point.getparent()
    for p in parent:
        if p.tag == pathlist[-1]:
            valuelist.append(p.attrib['value'])