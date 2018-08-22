import tensorflow as tf
import numpy as np

# An example use of tensorflows iteratable datasets
# Here we make a data set from an array of tensors.
# The iterator is used in the compute graph to pull each
# tensor from the data set and perform a calculation on it.
# We also add a print node into the graph allowing us to print
# the tensor values of an node at a point during the calculation.



BATCH_SIZE = 2
numGrids = 3
width = 2
height = 3

# create a placeholder to dynamically switch between batch sizes
batch_size = tf.placeholder(tf.int64)
# using a placeholder
x = tf.placeholder(tf.float32, shape=[None,2])
# A variable to hold a tensor storing possibly multiple new input tensors (grids).
inGrids = tf.placeholder(tf.float32, [numGrids, width, height], name='InputGrids')
# variable to store just one new input tensor (one grid).
newGrid = tf.placeholder(tf.float32, [1, width, height], name='NewGrid')
# A variable to remember values between calls to the session
# We set it to be non trainable since we are not using back propagation in this example.
prevGrid = tf.get_variable("prevGrid",
                           shape=(width, height),
                           dtype=tf.float32,
                           initializer=tf.zeros_initializer,
                           trainable=False)

dataset = tf.data.Dataset.from_tensor_slices(inGrids)

newInputMat = np.array(np.random.randint(10, size=(numGrids, width, height)))
print("newInputMat = \n%s" % newInputMat)


#data = np.random.sample((100,2))


iter = dataset.make_initializable_iterator() # create the iterator
#el = iter.get_next()

newGrid = iter.get_next()
#newGrid2 = iter.get_next()

# Calculate the inputs to each column
mult_y = tf.multiply(newGrid, newGrid)
# Print the output. Use a print node in the graph. The first input is the input data to pass to this node,
# the second is an array of which nodes in the graph you would like to print
#pr_mult = tf.Print(mult_y, [mult_y, newGrid], summarize = 25)
pr_mult = tf.Print(mult_y, [mult_y, newGrid], summarize = 25)

assignment = prevGrid.assign_add(pr_mult)

with tf.Session() as sess:
    # feed the placeholder with data
    sess.run(iter.initializer, feed_dict={ inGrids: newInputMat ,batch_size: BATCH_SIZE})
    #print(sess.run(newGrid2))
    #print(sess.run(tf.report_uninitialized_variables()))
    sess.run(prevGrid.initializer)
    for i in range(numGrids):
        print("New input Grid calculation")
        print(sess.run(assignment))
        # Set the output to the preGrid Value to store it. This is a tf.variable that will store the values.

        #print(sess.run(el))
        #sess.run(el)
