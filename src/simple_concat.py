
import numpy as np
import tensorflow as tf


#t_a = np.random.randint(3, size=(3))
t_a = np.arange(3)
t_b = np.random.randint(2, size=(3))

np_array_1 = np.random.randint(10, 20, size=(3, 2))
print("t_a = \n%s" % t_a)
print("t_b = \n%s" % t_b)
print("np_array_1 = \n%s" % np_array_1)

ta = tf.constant(t_a, dtype=tf.int32)
tfMat = tf.constant(np_array_1, dtype=tf.int32)

ind_1 = tf.stack([t_a, t_b], axis=-1)

pr_res = tf.Print(ind_1,
                  [ind_1],
                  message="Print", summarize=200)

minLocalAct = tf.gather_nd(tfMat, pr_res)
print("minLocalAct.shape = \n%s" % minLocalAct.shape)
#transMinLocalAct = tf.transpose(minLocalAct)
transMinLocalAct = tf.expand_dims(minLocalAct, 1)
print("transMinLocalAct.shape = \n%s" % transMinLocalAct.shape)
bdim = [transMinLocalAct.get_shape().as_list()[0], 2]
print("bdim = \n%s" % bdim)
minLocalActExpand = tf.broadcast_to(transMinLocalAct, bdim)
print("minLocalActExpand.shape = \n%s" % minLocalActExpand.get_shape().as_list())


with tf.Session() as sess:
    res = sess.run(minLocalActExpand)
    print(res)
