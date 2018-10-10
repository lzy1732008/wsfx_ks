# ssls = ['0.12//0.23//1.2//', '0.9//0.8//0.7 ']
# ls = [list(map(float,list(filter(lambda x:x.strip() != '', ss.split('//'))))) for ss in ssls]
# print(ls)
# import numpy as np
# s = [[0,1,2,3],[4,5,6,7]]
# print(np.mean(s,axis=0))

import tensorflow as tf
num = tf.constant([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]])
with tf.Session() as sess:
    t = sess.run(tf.reshape(num,shape=[3,]))
    print(sess.run(tf.reshape(t,shape=[2,3,2])))




