#simple_loop.py

import tensorflow as tf
import numpy as np
import cProfile

# Profiling function
def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func



def body(x):
    a = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100)
    b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    c = a + b
    return tf.nn.relu(x + c)


def condition(x):
    return tf.reduce_sum(x) < 1000

#@do_cprofile  # For profiling
def run_tf(x):


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = tf.while_loop(condition, body, [x])
        print(result.eval())

if __name__ == '__main__':
    x = tf.Variable(tf.constant(0, shape=[2, 2]))
    run_tf(x)


