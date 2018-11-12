import xlwt
import xlrd
import os
from xlutils.copy import copy


# rows:list
# colums:list
# data:list[list]
def createx(wsname, rows, colums, data ,dir):

    fnt = xlwt.Font()
    fnt.name = 'SimSun'
    fnt.bold = True
    fnt.height = 250
    wb = xlwt.Workbook()
    style = xlwt.easyxf('align: wrap on;')

    style.font = fnt

    ws = wb.add_sheet(wsname);


    #设置行
    for i in range(len(rows)):
        ws.write(i + 1, 0, rows[i] ,style)


    #设置列
    for i in range(len(colums)):
        ws.write(0, i + 1, colums[i] ,style)

    for i in range(len(colums)+1):
        ws.col(i).width = 256 * 40

    ws.panes_frozen = True
    ws.horz_split_pos = 1


    #录入数据
    for i in range(len(data)):
        for j in range(len(data[i])):
            ws.write(i+1,j+1,data[i][j])
    wb.save(dir+'/'+wsname+'_20181110.xls');



def getcolls(excelpath):
    excelfile = xlrd.open_workbook(excelpath)
    sheet = excelfile.sheet_by_index(0)
    rowls = []
    for i in range(1,sheet.ncols):
        # print(sheet.col_values(i))
        cell = sheet.cell_value(0,i)
        rowls.append(cell)
    return rowls

def getrow2ls(excelpath):
    excelfile = xlrd.open_workbook(excelpath)
    sheet = excelfile.sheet_by_index(0)
    colls = []
    for i in range(1,sheet.nrows):
        # print(sheet.col_values(i))
        cell = sheet.cell_value(i,1)
        colls.append(cell)
    return colls

def getrowls(excelpath):
    excelfile = xlrd.open_workbook(excelpath)
    sheet = excelfile.sheet_by_index(0)
    colls = []
    for i in range(1,sheet.nrows):
        # print(sheet.col_values(i))
        cell = sheet.cell_value(i,0)
        colls.append(cell)
    return colls



def getexceldata(excelpath):
    data = []
    excelfile = xlrd.open_workbook(excelpath)
    sheet = excelfile.sheet_by_index(0)
    # print(sheet.name)
    for i in range(1,sheet.ncols):
        col = []
        # print(sheet.col_values(i))
        for j in range(1,sheet.nrows):
            cell = sheet.cell_value(j,i)
            if isinstance(cell,float) == True:
                col.append(cell)
            if isinstance(cell,str) == True:
                if cell.strip() == '':
                    col.append(0)
                else:
                    col.append(int(cell[0]))
            # print(col)
        data.append(col)
    return data


def alterexcel(excelapath,cols,rows,datas):
    p= xlrd.open_workbook(excelapath)
    wb = copy(p)
    sheet = wb.get_sheet(0)
    print(sheet.name)
    pdata = zip(cols,rows,datas)
    for col,row,data in pdata:
        sheet.write(int(row.strip())+1,int(col.strip())+1,data)
    wb.save(excelapath)

