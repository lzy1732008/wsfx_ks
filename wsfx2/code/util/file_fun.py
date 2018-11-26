def write_dict(data,filepath):
    f = open(filepath, 'w' ,encoding='utf-8')
    all_s = ''
    for key in data: all_s += key + "|" + str(data[key]) +"\n"
    f.write(all_s)

def getlines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().split('\n')
    lines = list(filter(lambda x: str(x).strip() != '', content))
    return lines

def countlines(filepath):
    f = open(filepath,'r',encoding='utf-8')
    lines = f.read().split('\n')
    print(len(lines))

data_dir = '../../source/dataset/set-lyw'
trainpath = data_dir+'/trainlyw.txt'
t_f = open(trainpath,'r',encoding='utf-8')
lines = t_f.read().split('\n')
facts = lines[9].split('|')[1]
fs = facts.split(' ')
vect = fs[0].split('//')
vs = list(filter(lambda x: str(x).strip() != '', vect))
print(len(vs))






