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


import numpy as np
# s = [[1,2,3],[4,5,6]]
# print(s.count([1,2,3]))


# import tensorflow.contrib.keras as kr
# s1=[[1,2,3,4,5]]
# s2=[[1,2]]
# print(kr.preprocessing.sequence.pad_sequences(s1,3))
# print(kr.preprocessing.sequence.pad_sequences(s2,3))

# start = np.random.randn(128)
# print(start.dtype)

# import tensorflow as tf
# sess = tf.InteractiveSession()
# a = tf.placeholder(dtype=tf.float32,name='a')
# t = a + 1
# feed_dict = {a:1.0}
# print(t.eval(feed_dict=feed_dict))

# b = tf.constant([[0,0,0],[1,2,3],[4,5,6],[0,0,0]],name='b')
# len = tf.shape(b)[1]
# print(len)
# for _ in range(1,(b.shape)[0]-1):
#     s.append(b[_-1] + b[_+1])
# news = tf.convert_to_tensor(s)
# print(sess.run(news))



# import matplotlib.pyplot as plt
# import numpy as np
# a = np.linspace(0,4,16).reshape(4,4)
# print(a)
# plt.imshow(a,interpolation='nearest', cmap='bone',origin='lower')
# plt.colorbar()
# plt.xticks()
# plt.yticks()
# plt.show()


import numpy as np
# def discosin(v1,v2):
#     num = np.sum(v1 * v2)
#     denom = np.linalg.norm(v1) * np.linalg.norm(v2)
#     cos = num/denom
#     sim = 0.5 + 0.5 * cos
#     return sim
#
# v1 = np.array([1,2,3,4])
# v2 = np.array([2,3,4,5])
# print(discosin(v1,v2))












