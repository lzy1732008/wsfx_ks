from wsfx2.code.util.ws_fun import getFTList
from wsfx2.code.util.file_fun import write_dict
from wsfx2.code.util.sql_fun import get_LawQW,connectSQL
import os

#统计数据集有哪些法条以及这些法条的内容
'''
输入：dict_path:文书数据集的路径
输出：一个dict[ftname,ftzw]
'''
def tj_ft_zw(dict_path):
   ft_zw = {}
   dir = os.listdir(dict_path)
   for wsname in dir:
       if not wsname.endswith(".xml"): continue
       ftls,nrls = getFTList(os.path.join(dict_path,wsname))
       for i in range(len(ftls)):
           if not ft_zw.get(ftls[i]): ft_zw[ftls[i]] = nrls[i]
   return ft_zw

#统计数据集中所有的法条及其正文
# dict_path = "/Users/wenny/nju/task/法条文书分析/2014filled/2014"
# txt_path = '../../source/other/ft_nr.txt'
# p = tj_ft_zw(dict_path)
# write_dict(p, txt_path)

#统计数据集中所有的法律
'''
输入:dict_path 数据集xml文件的路径
输出:fts 法律列表 list类型
'''
def tf_ft(dict_path):
    fts = []
    dir = os.listdir(dict_path)
    for wsname in dir:
        if not wsname.endswith(".xml"): continue
        ftls, nrls = getFTList(os.path.join(dict_path, wsname))
        for ft in ftls:
            print(ft)
            ft = ft[:str(ft).find('(')]
            print(ft)
            if ft not in fts: fts.append(ft)
    return fts

#统计数据集中所有的法律
# dict_path = "/Users/wenny/nju/task/法条文书分析/2014filled/2014"
# txt_path = '../../source/other/ft_name.txt'
# fts = tf_ft(dict_path)
# f = open(txt_path,'w',encoding='utf-8')
# f.write('\n'.join(fts))

#根据txt文件中的法律名称去数据库中找到全文并存储到txt中
'''
输入：sourcepath:存放所有法律名的txt文件
     targetpath:存放法律全文的目录
'''
def store_LawQW(sourcepath, targetdict):
    f = open(sourcepath, 'r', encoding='utf-8')
    lines = f.read().split('\n')
    fts = list(filter(lambda x: x.strip()!='',lines))
    cursor = connectSQL()
    for ft in fts:
        ft_qw = get_LawQW(cursor=cursor, law_name = ft)
        if ft_qw!='': open(targetdict+'/'+ft+'.txt','w',encoding='utf-8').write(ft_qw)
        else: print(ft)

# store_LawQW('../../source/法条/allft.txt','../../source/法律全文')










