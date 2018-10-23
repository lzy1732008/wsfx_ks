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


import tensorflow as tf
# import pandas as pd
# num = np.array([[[1,2,3],[4,5,6],[1,1,1]],[[7,8,9],[10,11,12],[2,2,2]]])
# for n in num:
#     t_n = pd.DataFrame(n)
#     sum = (np.array(t_n.shift()+n)).tolist()
#     sum.append([0]*len(n[0]))
#     print(sum[1:])

# num = tf.constant([[[1,2,3],[4,5,6],[1,1,1],[1,1,1]],[[7,8,9],[10,11,12],[2,2,2],[1,1,1]]])
# num2 = tf.constant([[[1,2,3],[4,5,6],[1,1,1],[1,1,1]],[[7,8,9],[10,11,12],[2,2,2],[1,1,1]]])
# r1 = tf.keras.backend.repeat_elements(num,2,axis=2)
# r2 = tf.reshape(r1,shape=[2,4,2,3])
# with tf.Session() as sess:
#       print(sess.run(num * num2))
#     print(sess.run(r1))
# #     print(r1.shape)
#     print(sess.run(r2))









