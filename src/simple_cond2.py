import tensorflow as tf
import numpy as np

x = tf.constant(0.5)
y = tf.constant(0.5)

np_array_1 = np.random.rand(3, 2)
np_array_2 = np.random.rand(3, 2)

print("arr 1 = \n%s" % np_array_1)
print("arr 2 = \n%s" % np_array_2)


def f1(): return tf.multiply(x, 17)


def f2(): return tf.add(y, 23)
#r = tf.cond(tf.less(np_array_1, np_array_2), f1, f2)

#c1 = tf.less(np_array_1, np_array_2)


c2 = tf.cast(tf.greater(tf.cast(np_array_1,tf.float32),
                        tf.cast(np_array_2, tf.float32)),
             tf.float32)
# r is set to f1().
# Operations in f2 (e.g., tf.add) are not executed.
# Start training
with tf.Session() as sess:


    print(c2.eval())