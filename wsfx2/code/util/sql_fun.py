import pymysql


def connectSQL():
    connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='laws', db='laws', charset='utf8mb4')
    # 通过cursor创建游标
    cursor = connection.cursor()
    return cursor

def get_LawQW(cursor, law_name):
    sql = "select DOC_TEXT from law_0_doc where DOC_name='" + law_name+"'"
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
        return result[0]
    except:
        return ''
#
# cursor = connectSQL()
# sql = "select DOC_NAME from law_0_doc"
# cursor.execute(sql)
# result = cursor.fetchall()
# f = open('../../source/allft.txt','w',encoding='utf-8')
# for ft in result:
#     f.write(ft[0]+'\n')
#
# f.close()






