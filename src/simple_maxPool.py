import tensorflow as tf

temp = [0., 0., 1., 0., 0., 0., 1.5, 2.5]

# Reshape the tensor to be 5 dimensions.
values = tf.reshape(temp, [1, 1, 2, 2, 2])

# Use an averaging pool on the tensor.
p_avg = tf.nn.pool(input=values,
    window_shape=[2, 2, 2],
    pooling_type="AVG",
    padding="SAME")

# Use max with this pool.
p_max = tf.nn.pool(input=values,
    window_shape=[2, 2, 2],
    pooling_type="MAX",
    padding="SAME")

session = tf.Session()

# Print our tensors.
print("VALUES")
print(session.run(values))
print("POOL")
print(session.run(p_avg))
print("POOL MAX")
print(session.run(p_max))