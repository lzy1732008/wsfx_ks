# ssls = ['0.12//0.23//1.2//', '0.9//0.8//0.7 ']
# ls = [list(map(float,list(filter(lambda x:x.strip() != '', ss.split('//'))))) for ss in ssls]
# print(ls)
# import numpy as np
# s = [[0,1,2,3],[4,5,6,7]]
# print(np.mean(s,axis=0))

# import tensorflow as tf
# num = tf.constant([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]])
# with tf.Session() as sess:
#     t = sess.run(tf.reshape(num,shape=[3,]))
#     print(sess.run(tf.reshape(t,shape=[2,3,2])))

import numpy as np
# import pandas as pd
# num = np.array([[[1,2,3],[4,5,6],[1,1,1]],[[7,8,9],[10,11,12],[2,2,2]]])
# for n in num:
#     t_n = pd.DataFrame(n)
#     sum = (np.array(t_n.shift()+n)).tolist()
#     sum.append([0]*len(n[0]))
#     print(sum[1:])

num = np.array([[[1,2,3],[4,5,6],[1,1,1],[1,1,1]],[[7,8,9],[10,11,12],[2,2,2],[1,1,1]]])
from wsfx2.code.train.loader import data_ngram
print(data_ngram(num))








