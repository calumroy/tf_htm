from __future__ import print_function
import tensorflow as tf
import numpy as np
from numpy import array, arange, ix_
from datetime import datetime
import random

# Create a tensorflow for calculating
# for each column a list of columns which that column can
# be inhibited by. Set the winning columns in this list as one.
# If a column is inhibited already then all those positions in
# the colwinners relating to that col are set as one. This means
# the inhibited columns don't determine the active columns


# Create the theano function for calculating
# a matrix of the columns which should stay active because they
# won the inhibition convolution for all columns.
# if a column is inhibited then set that location to one only if
# that row does not represent that inhibited column.
# col_pat = T.matrix(dtype='int32')
# act_cols = T.matrix(dtype='float32')
# col_num2 = T.matrix(dtype='int32')
# row_numMat4 = T.matrix(dtype='int32')
# cur_inhib_cols4 = T.vector(dtype='int32')

# test_meInhib = T.switch(T.eq(cur_inhib_cols4[row_numMat4], 1), 0, 1)
# set_winners = act_cols[col_pat-1, col_num2]
# check_colNotInhib = T.switch(T.lt(cur_inhib_cols4[col_pat-1], 1), set_winners, test_meInhib)
# check_colNotPad = T.switch(T.ge(col_pat-1, 0), check_colNotInhib, 0)
# get_activeColMat = function([act_cols,
#                                   col_pat,
#                                   col_num2,
#                                   row_numMat4,
#                                   cur_inhib_cols4],
#                                  check_colNotPad,
#                                  on_unused_input='warn',
#                                  allow_input_downcast=True
#                                  )


def calculateNonPaddingSumVect(colInConvoleList):
    # Store a vector where each element stores for a column how many times
    # that column appears in other columns convole lists.
    gtZeroMat = tf.greater(colInConvoleList, 0)
    sumRowMat = tf.reduce_sum(
                              tf.cast(gtZeroMat, tf.float32),
                              1)

    # Run the tf graph
    with tf.Session() as sess:
        outputGrid = sess.run(sumRowMat)

    print("nonPaddingSumVect = \n%s" % outputGrid)
    return outputGrid


def calculateActiveColumnVect(activeCols, inhibCols, colOverlapVect):
    # Calculate for each column a list of columns which that column can
    # be inhibited by. Set the winning columns in this list as one.
    # If a column is inhibited already then all those positions in
    # the colwinners relating to that col are set as one. This means
    # the inhibited columns don't determine the active columns

    # Calculate
    # a matrix of the columns which should stay active because they
    # won the inhibition convolution for all columns.
    # if a column is inhibited then set that location to one only if
    # that row does not represent that inhibited column.
    with tf.name_scope('get_activeColMat'):
        act_cols = tf.convert_to_tensor(activeCols, dtype=tf.int32, name='act_cols')
        col_pat = tf.convert_to_tensor(colInConvoleList, dtype=tf.int32, name='col_pat')
        col_num2 = tf.convert_to_tensor(col_num, dtype=tf.int32, name='col_num2')
        row_numMat4 = tf.convert_to_tensor(row_numMat, dtype=tf.int32, name='row_numMat4')
        inhib_cols2 = tf.convert_to_tensor(inhibCols, dtype=tf.int32, name='inhib_cols2')

        cur_inhib_cols_row = tf.gather(inhib_cols2, row_numMat4, name='cur_inhib_cols_row')
        col_pat_neg_one = tf.add(col_pat, -1, name='col_pat_neg_one')

        zeros_mat = tf.zeros_like(cur_inhib_cols_row, dtype=tf.int32, name='zeros_mat')
        ones_mat = tf.ones_like(cur_inhib_cols_row, dtype=tf.int32, name='ones_mat')

        test_meInhib = tf.where(tf.equal(cur_inhib_cols_row, 1), zeros_mat, ones_mat, name='test_meInhib')
        indicies = tf.stack([tf.maximum(col_pat_neg_one, zeros_mat), col_num2], axis=-1)
        # For each columns colInConvole list check which ones are active.
        set_winners = tf.gather_nd(act_cols, indicies, name='set_winners')

        # Get the values at the non negative indicies.
        # We get the value at index zero for the negative indicies. These are not used.
        cur_inhib_col_pat = tf.gather(inhib_cols2, tf.maximum(col_pat_neg_one, zeros_mat), name='cur_inhib_col_pat')

        check_colNotInhib = tf.where(tf.less(cur_inhib_col_pat, ones_mat), set_winners, test_meInhib, name='check_colNotInhib')
        check_colNotPad = tf.where(tf.greater_equal(col_pat_neg_one, zeros_mat), check_colNotInhib, zeros_mat)

        colwinners = check_colNotPad

    # Now calculate which columns won enough of their colwinners list.
    # This creates a vector telling us which columns have the highest
    # overlap values and should be active.
    # The rows that have more then or equal to
    # the input nonPaddingSumVect. If this is true then set
    # in the output vector the col this row represents as active.
    # This function calculates if a column beat all the other non inhibited
    # columns in the convole overlap groups.
    with tf.name_scope('get_activeColVect'):
        # Make sure the self.nonPaddingSumVect is greater than zero.
        zeros_vec = tf.zeros_like(nonPaddingSumVect, dtype=tf.int32, name='zeros_vec')
        w_cols = tf.reduce_sum(colwinners, axis=1)
        test_lcol = tf.cast(tf.greater_equal(w_cols, nonPaddingSumVect), tf.int32, name='test_lcol')
        test_meInhib2 = tf.where(tf.greater(nonPaddingSumVect, 0), test_lcol, zeros_vec, name='test_meInhib2')
        # If the column has a zero overlap value (ie its overlap value
        # plus the tiebreaker is less then one then do not allow it to be active.
        activeColumnVect = tf.where(tf.less(colOverlapVect, 1), zeros_vec, test_meInhib2)

        # Print the output.
        with tf.name_scope('print'):
            # Use a print node in the graph. The first input is the input data to pass to this node,
            # the second is an array of which nodes in the graph you would like to print
            # pr_mult = tf.Print(mult_y, [mult_y, newGrid], summarize = 25)
            print_out = tf.print("\n Printing Tensors \n"
                                 "\n col_pat_neg_one = \n", col_pat_neg_one,
                                 "\n colOverlapVect = \n", colOverlapVect,
                                 "\n activeColumnVect = \n", activeColumnVect,
                                 "\n cur_inhib_cols_row = \n", cur_inhib_cols_row,
                                 "\n test_meInhib = \n", test_meInhib,
                                 "\n act_cols = \n", act_cols,
                                 "\n indicies = \n", indicies,
                                 "\n set_winners = \n", set_winners,
                                 "\n cur_inhib_col_pat = \n", cur_inhib_col_pat,
                                 "\n check_colNotInhib = \n", check_colNotInhib,
                                 "\n colwinners = \n", colwinners,
                                 "\n colInConvoleList = \n", colInConvoleList,
                                 "\n w_cols = \n", w_cols,
                                 "\n nonPaddingSumVect = \n", nonPaddingSumVect,
                                 "\n test_lcol = \n", test_lcol,
                                 "\n activeColumnVect = \n", activeColumnVect,
                                 "\n inhibCols = \n", inhibCols,
                                 summarize=200)

        # Make sure the print_out is performed during the graph execution.
        with tf.control_dependencies([print_out]):
            # Perform some tf operation so the print out occurs.
            activeColumnVectFinal = tf.multiply(activeColumnVect, 1)



    #print("activeCols = \n%s" % activeCols)

    return activeColumnVectFinal

# colInConvoleList = (
#                     [[5, 4, 2, 1],
#                      [6, 5, 3, 2],
#                      [0, 6, 0, 3],
#                      [8, 7, 5, 4],
#                      [9, 8, 6, 5],
#                      [0, 9, 0, 6],
#                      [0, 0, 8, 7],
#                      [0, 0, 9, 8],
#                      [0, 0, 0, 9]])

# potentialWidth = 2
# potentialHeight = 2
# width = 3
# height = 3

colInConvoleList =  ([[ 6,  5,  0,  2,  1,  0,  0,  0,  0],
       [ 7,  6,  5,  3,  2,  1,  0,  0,  0],
       [ 8,  7,  6,  4,  3,  2,  0,  0,  0],
       [ 0,  8,  7,  0,  4,  3,  0,  0,  0],
       [10,  9,  0,  6,  5,  0,  2,  1,  0],
       [11, 10,  9,  7,  6,  5,  3,  2,  1],
       [12, 11, 10,  8,  7,  6,  4,  3,  2],
       [ 0, 12, 11,  0,  8,  7,  0,  4,  3],
       [14, 13,  0, 10,  9,  0,  6,  5,  0],
       [15, 14, 13, 11, 10,  9,  7,  6,  5],
       [16, 15, 14, 12, 11, 10,  8,  7,  6],
       [ 0, 16, 15,  0, 12, 11,  0,  8,  7],
       [ 0,  0,  0, 14, 13,  0, 10,  9,  0],
       [ 0,  0,  0, 15, 14, 13, 11, 10,  9],
       [ 0,  0,  0, 16, 15, 14, 12, 11, 10],
       [ 0,  0,  0,  0, 16, 15,  0, 12, 11]])
potentialWidth = 3
potentialHeight = 3
width = 4
height = 4

# Create a matrix that just holds the column number for each element
col_num = np.array([[i for i in range(potentialWidth*potentialHeight)]
                    for j in range(width*height)])

# Create a matrix that just holds the row number for each element
row_numMat = np.array([[j for i in range(potentialWidth*potentialHeight)]
                      for j in range(width*height)])


# activeCols = np.random.randint(2, size=(9, 4))
# activeCols = np.array(
#                       [[0, 0, 0, 0],
#                        [1, 1, 1, 0],
#                        [0, 0, 0, 1],
#                        [0, 1, 0, 0],
#                        [0, 0, 0, 0],
#                        [0, 1, 0, 1],
#                        [1, 0, 1, 0],
#                        [1, 0, 0, 0],
#                        [0, 0, 1, 1]])


activeCols = (
            [[0, 0, 0, 0, 1, 0, 0, 1, 0],
             [0, 0, 0, 1, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 1],
             [0, 0, 0, 0, 1, 0, 0, 1, 0],
             [0, 1, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 1, 0, 0, 0, 0],
             [0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 1, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 1, 0, 0, 0, 0]])


print("activeCols = \n%s" % activeCols)
# Create just a vector storing if a column is inhibited or not
# inhibCols = np.array([random.randint(0, 1) for i in range(width*height)])
# inhibCols = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
inhibCols = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#inhibCols = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1])

print("inhibCols = \n%s" % inhibCols)


# colOverlapMat = np.array([[8, 4, 5, 8],
#                           [8, 6, 1, 6],
#                           [7, 7, 9, 4],
#                           [2, 3, 1, 5]])
# colOverlapMat = np.random.randint(10, size=(3, 3))
# colOverlapMat = np.array(
#                          [[8, 8, 8],
#                           [0, 9, 0],
#                           [0, 0, 0]])



#print("colOverlapMat = \n%s" % colOverlapMat)

# tieBreaker = ([[0.1, 0.2, 0.3],
#               [0.4, 0.5, 0.6],
#               [0.7, 0.8, 0.9]])

# overlapsGridTie = colOverlapMat + tieBreaker
# Create a vector of the overlap values for each column
# colOverlapVect = overlapsGridTie.flatten()

#colOverlapVect = np.array([9.05882359, 0.117647059, 0.176470593, 0.235294119, 0.294117659, 0.352941185, 0.411764711, 0.470588237, 0.529411793, 0.588235319, 9.64705849, 0.70588237, 0.764705896, 0.823529422, 0.882352948, 0.941176474])
colOverlapVect = np.array([8.05882359, 4.11764717, 5.17647076, 8.23529434, 8.29411793, 6.35294104, 1.41176474, 6.47058821, 7.52941179, 7.58823538, 9.64705849, 4.70588255, 2.7647059, 3.82352948, 1.88235295, 5.94117641])

# Create the NonPaddingSumVect
nonPaddingSumVect = calculateNonPaddingSumVect(colInConvoleList)

activeColVect = calculateActiveColumnVect(activeCols, inhibCols, colOverlapVect)

logsPath = '/tmp/tensorflow_logs/example/tf_htm/'
now = datetime.now()
logsPath = logsPath + now.strftime("%Y%m%d-%H%M%S") + "/"
summary_writer = tf.summary.FileWriter(logsPath, graph=tf.get_default_graph())



# Start training
with tf.Session() as sess:
    print("Run Sess")
    tf_activeColVect = activeColVect.eval()
    print("activeColVect = \n", tf_activeColVect)
