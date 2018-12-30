#simple_loop.py

import tensorflow as tf
import numpy as np
from datetime import datetime
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



def body(x, y):
    a = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100)
    b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    c = a + b

    d = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=10000)

    return tf.nn.relu(x + c), (y + d)


def condition(x, y):
    return tf.reduce_sum(x) < 10000

#@do_cprofile  # For profiling
def run_tf(x, y):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result1, result2 = tf.while_loop(condition, body, [x, y], back_prop=False)

        result = result1 + result2
        print(result.eval())



        summary_writer = tf.summary.FileWriter(logsPath, graph=tf.get_default_graph())


if __name__ == '__main__':
    # Set the location to store the tensorflow logs so we can run tensorboard.
    logsPath = '/tmp/tensorflow_logs/example/tf_htm/'
    now = datetime.now()
    logsPath = logsPath + now.strftime("%Y%m%d-%H%M%S") + "/"

    x = tf.Variable(tf.constant(0, shape=[2, 2]))
    y = tf.Variable(tf.constant(0, shape=[2, 2]))
    # x = tf.zeros(shape=[2, 2], dtype=tf.int32)
    # y = tf.zeros(shape=[2, 2], dtype=tf.int32)
    # x = tf.convert_to_tensor([[0, 0], [0, 0]], dtype=tf.int32)
    # y = tf.convert_to_tensor([[0, 0], [0, 0]], dtype=tf.int32)
    # x = tf.constant(0, shape=[2, 2])
    # y = tf.constant(0, shape=[2, 2])

    run_tf(x, y)


