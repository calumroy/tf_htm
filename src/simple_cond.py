'''add ta and tb 5 times, and then multiply them 5 times'''

import tensorflow as tf

ta = tf.ones([3], dtype=tf.float32)
tb = tf.zeros([3], dtype=tf.float32)
global_step = tf.Variable(0, trainable=False)
add_global_step = global_step.assign_add(1)
add_step = 5
total_step = 10

cond_op = tf.cond(tf.less(global_step, add_step), lambda:tf.add(ta, tb), lambda: ta*tb)

sv = tf.train.Supervisor(global_step=global_step)
with sv.managed_session() as sess:
    for i in xrange(total_step):
        res = sess.run(cond_op)
        print(i, res)
        sess.run(add_global_step)