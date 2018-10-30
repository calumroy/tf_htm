import tensorflow as tf
import numpy as np
from numpy import array, arange, ix_
from datetime import datetime

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
# self.col_pat = T.matrix(dtype='int32')
# self.act_cols = T.matrix(dtype='float32')
# self.col_num2 = T.matrix(dtype='int32')
# self.row_numMat4 = T.matrix(dtype='int32')
# self.cur_inhib_cols4 = T.vector(dtype='int32')

# self.test_meInhib = T.switch(T.eq(self.cur_inhib_cols4[self.row_numMat4], 1), 0, 1)
# self.set_winners = self.act_cols[self.col_pat-1, self.col_num2]
# self.check_colNotInhib = T.switch(T.lt(self.cur_inhib_cols4[self.col_pat-1], 1), self.set_winners, self.test_meInhib)
# self.check_colNotPad = T.switch(T.ge(self.col_pat-1, 0), self.check_colNotInhib, 0)
# self.get_activeColMat = function([self.act_cols,
#                                   self.col_pat,
#                                   self.col_num2,
#                                   self.row_numMat4,
#                                   self.cur_inhib_cols4],
#                                  self.check_colNotPad,
#                                  on_unused_input='warn',
#                                  allow_input_downcast=True
#                                  )





def calculateActiveColumnVect(activeCols, inhibCols, colOverlapVect):
    # Calculate for each column a list of columns which that column can
    # be inhibited by. Set the winning columns in this list as one.
    # If a column is inhibited already then all those positions in
    # the colwinners relating to that col are set as one. This means
    # the inhibited columns don't determine the active columns
    # colwinners = self.get_activeColMat(activeCols,
    #                                    colInConvoleList,
    #                                    col_num,
    #                                    row_numMat,
    #                                    inhibCols)

    #test_meInhib = tf.cond(tf.eq(cur_inhib_cols4[row_numMat4], 1), 0, 1)
    #set_winners = self.act_cols[self.col_pat-1, self.col_num2]

    act_cols = tf.cast(activeCols, dtype=tf.int32)
    col_pat = tf.cast(colInConvoleList, dtype=tf.int32)
    col_num2 = tf.cast(col_num, dtype=tf.int32)
    row_numMat4 = tf.cast(row_numMat, dtype=tf.int32)
    cur_inhib_cols4 = tf.cast(inhibCols, dtype=tf.int32)

    cur_inhib_cols_row = tf.gather(cur_inhib_cols4, row_numMat4, name='cur_inhib_cols_row')
    col_pat_neg_one = tf.add(col_pat, -1, name='col_pat_neg_one')

    zeros_mat = tf.zeros_like(cur_inhib_cols_row, name='zeros_mat')
    ones_mat = tf.ones_like(cur_inhib_cols_row, name='ones_mat')

    test_meInhib = tf.where(tf.equal(cur_inhib_cols_row, 1), zeros_mat, ones_mat, name='test_meInhib')
    indicies = tf.stack([tf.maximum(col_pat_neg_one, zeros_mat), col_num2], axis=-1)
    set_winners = tf.gather_nd(act_cols, indicies, name='set_winners')
    # set_winners = self.act_cols[col_pat_neg_one, col_num2]

    # Get the values at the non negative indicies. We get the value at index zero for the negative indicies. These are not used.
    cur_inhib_col_pat = tf.gather(cur_inhib_cols4, tf.maximum(col_pat_neg_one, zeros_mat), name='cur_inhib_col_pat')

    check_colNotInhib = tf.where(tf.less(cur_inhib_col_pat, ones_mat), set_winners, test_meInhib, name='check_colNotInhib')
    check_colNotPad = tf.where(tf.greater(col_pat_neg_one, zeros_mat), check_colNotInhib, zeros_mat)


    # tf.multiply(
    #             tf.cast(tf.greater(rolledInputGrid, 0), tf.float32),
    #             tf.cast(rolledInputGrid +
    #                     tf.gather(firstNonZeroIndVect, self.col_num) -
    #                     tf.gather(firstNonZeroVect, self.col_num) + 1,
    #                     tf.float32)
    #             )

    # nonPadOrInhibSumVect = self.get_gtZeroMat(nonPadOrInhibSumVect)
    # nonPadOrInhibSumVect = self.get_sumRowMat(nonPadOrInhibSumVect)
    # print "nonPadOrInhibSumVect = \n%s" % nonPadOrInhibSumVect

    # print "self.nonPaddingSumVect = \n%s" % self.nonPaddingSumVect
    # print "self.colInConvoleList = \n%s" % self.colInConvoleList
    # print "inhibCols = \n%s" % inhibCols
    # print "colwinners = \n%s" % colwinners

    # Now calculate which columns won enough of their colwinners list.
    # This creates a vector telling us which columns have the highest
    # overlap values and should be active.
    # Make sure the self.nonPaddingSumVect is greater than zero.

    #activeColumnVect = get_activeColVect(colwinners, self.nonPaddingSumVect)

    # If the column has a zero overlap value (ie its overlap value
    # plus the tiebreaker is less then one then do not allow it to be active.
    #activeColumnVect = self.disable_zeroOverlap(colOverlapVect,
    #                                            activeColumnVect)
    # print "activeColumnVect = \n%s" % activeColumnVect



    # Print the output.
    with tf.name_scope('print'):
        # Use a print node in the graph. The first input is the input data to pass to this node,
        # the second is an array of which nodes in the graph you would like to print
        # pr_mult = tf.Print(mult_y, [mult_y, newGrid], summarize = 25)
        activeColumnVect = tf.Print(check_colNotPad,
                                   [check_colNotPad],
                                   message="Print", summarize=200)



    print("activeCols = \n%s" % activeCols)

    return activeColumnVect

colInConvoleList = (
                    [[5, 4, 2, 1],
                     [6, 5, 3, 2],
                     [0, 6, 0, 3],
                     [8, 7, 5, 4],
                     [9, 8, 6, 5],
                     [0, 9, 0, 6],
                     [0, 0, 8, 7],
                     [0, 0, 9, 8],
                     [0, 0, 0, 9]])

potentialWidth = 2
potentialHeight = 2
width = 3
height = 3

# Create a matrix that just holds the column number for each element
col_num = np.array([[i for i in range(potentialWidth*potentialHeight)]
                    for j in range(width*height)])

# Create a matrix that just holds the row number for each element
row_numMat = np.array([[j for i in range(potentialWidth*potentialHeight)]
                      for j in range(width*height)])


activeCols = np.random.randint(2, size=(9, 4))
print("activeCols = \n%s" % activeCols)
# Create just a vector storing if a column is inhibited or not
inhibCols = np.array([0 for i in range(width*height)])
print("inhibCols = \n%s" % inhibCols)
colOverlapMat = np.random.randint(10, size=(3, 3))
print("colOverlapMat = \n%s" % colOverlapMat)

tieBreaker = ([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9]])

overlapsGridTie = colOverlapMat + tieBreaker
# Create a vector of the overlap values for each column
colOverlapVect = overlapsGridTie.flatten()


activeColVect = calculateActiveColumnVect(activeCols, inhibCols, colOverlapVect)

logsPath = '/tmp/tensorflow_logs/example/tf_htm/'
now = datetime.now()
logsPath = logsPath + now.strftime("%Y%m%d-%H%M%S") + "/"
summary_writer = tf.summary.FileWriter(logsPath, graph=tf.get_default_graph())



# Start training
with tf.Session() as sess:
    print("Run Sess")
    tf_activeColVect = activeColVect.eval()
    print(tf_activeColVect)


# np_activeCol = calculateActiveCol(colOverlapMat)
# print("numpy activeCol = \n%s" % np_activeCol)
