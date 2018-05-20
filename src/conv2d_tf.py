import tensorflow as tf

# This shows an example of using the convolve 2d tensorflow function and how to pad the input
# and adjyst the kernel

'''
t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[1, 1,], [2, 2]])
# 'constant_values' is 0.
# rank of 't' is 2.
tf.pad(t, paddings, "CONSTANT")  # [[0, 0, 0, 0, 0, 0, 0],
                                 #  [0, 0, 1, 2, 3, 0, 0],
                                 #  [0, 0, 4, 5, 6, 0, 0],
                                 #  [0, 0, 0, 0, 0, 0, 0]]

tf.pad(t, paddings, "REFLECT")  # [[6, 5, 4, 5, 6, 5, 4],
                                #  [3, 2, 1, 2, 3, 2, 1],
                                #  [6, 5, 4, 5, 6, 5, 4],
                                #  [3, 2, 1, 2, 3, 2, 1]]

tf.pad(t, paddings, "SYMMETRIC")  # [[2, 1, 1, 2, 3, 3, 2],
                                  #  [2, 1, 1, 2, 3, 3, 2],
                                  #  [5, 4, 4, 5, 6, 6, 5],
                                  #  [5, 4, 4, 5, 6, 6, 5]]
'''



k = tf.constant([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=tf.float32, name='k')

i = tf.constant([
    [1, 2, 1, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 1, 0, 2]
], dtype=tf.float32, name='i')

s = tf.constant(
    [1, 1, 1, 1]
, dtype=tf.float32, name='s')

kernel = tf.reshape(k, [3, 3, 1, 1], name='kernel')
image  = tf.reshape(i, [1, 4, 4, 1], name='image')
stride  = [1,1,1,1]

# We will add our own padding so we can reflect the input.
# This prevents the edges from being unfairly hindered due to a smaller overlap with the kernel.
paddings = tf.constant([[0, 0,], [1, 1,], [1, 1], [0, 0,]])
padded_input = tf.pad(image, paddings, "REFLECT")

# tf.squeeze Removes dimensions of size 1 from the shape of a tensor.
res = tf.squeeze(tf.nn.conv2d(padded_input, kernel, stride, "VALID"))
# VALID means no padding
# "SAME" means use padding

logs_path = '/tmp/tensorflow_logs/example/tf_htm/'

with tf.Session() as sess:
	#sess.run(res)
   	session =  sess.run(res)
   	print(session)
   #print(image.eval())

   # op to write logs to Tensorboard
	summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

	# Write logs at every iteration
	#summary_writer.add_summary(summary, epoch * total_batch + i)

	print("\nRun the command line:\n" \
	"--> tensorboard --logdir=/tmp/tensorflow_logs " \
	"\nThen open http://0.0.0.0:6006/ into your web browser")
