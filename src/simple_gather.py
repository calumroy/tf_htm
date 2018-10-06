import numpy as np
import tensorflow as tf

x = np.asarray([1,2,3,3,2,5,6,7,1,3])
e = np.asarray([0,1,0,1,1,1,0,1,0,0])
x_t = tf.constant(x)
e_t = tf.constant(e)

#result = x_t * tf.gather(e_t, x_t)
result = tf.gather(x_t, e_t)

with tf.Session() as sess:
    print sess.run(result)  # ==> 'array([1, 0, 3, 3, 0, 5, 0, 7, 1, 3])'


# Gather with a condition
# tf.cond(tf.less(np_array_1, np_array_2), f1, f2)

result2 = tf.multiply(
                      tf.cast(tf.less(e_t, x_t), tf.float32),
                      tf.cast(e_t + tf.gather(x_t, e_t) - tf.gather(e_t, x_t), tf.float32)
                     )

with tf.Session() as sess:
    print(sess.run(result2))  # ==> 'array([0. 3. 0. 2. 3. 2. 1. 2. 0. 0.])''
