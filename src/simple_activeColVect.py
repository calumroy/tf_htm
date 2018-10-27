import tensorflow as tf
import numpy as np
from numpy import array, arange, ix_

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
    test_meInhib = 0
    set_winners = 0
    act_cols = activeCols
    col_pat = colInConvoleList
    col_num2 = col_num
    row_numMat4 = row_numMat
    cur_inhib_cols4 = inhibCols

    check_colNotInhib = tf.cond(tf.less(cur_inhib_cols4[col_pat-1], 1), set_winners, test_meInhib)
    check_colNotPad = tf.cond(tf.greater(col_pat-1, 0), check_colNotInhib, 0)


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
    activeColumnVect = self.get_activeColVect(colwinners, self.nonPaddingSumVect)

    # If the column has a zero overlap value (ie its overlap value
    # plus the tiebreaker is less then one then do not allow it to be active.
    activeColumnVect = self.disable_zeroOverlap(colOverlapVect,
                                                activeColumnVect)
    # print "activeColumnVect = \n%s" % activeColumnVect

    return activeColumnVect



colInConvoleList = (
[[ 6,  5,  2,  1],
 [ 7,  6,  3,  2],
 [ 8,  7,  4,  3],
 [ 0,  8,  0,  4],
 [10,  9,  6,  5],
 [11, 10,  7,  6],
 [12, 11,  8,  7],
 [ 0, 12,  0,  8],
 [14, 13, 10,  9],
 [15, 14, 11, 10],
 [16, 15, 12, 11],
 [ 0, 16,  0, 12],
 [ 0,  0, 14, 13],
 [ 0,  0, 15, 14],
 [ 0,  0, 16, 15],
 [ 0,  0,  0, 16]])

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

colOverlapMat = np.random.randint(10, size=(3, 3))
print("colOverlapMat = \n%s" % colOverlapMat)

tieBreaker = ([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9]])

overlapsGridTie = colOverlapMat + tieBreaker
# Create a vector of the overlap values for each column
colOverlapVect = overlapsGridTie.flatten()


activeColVect = calculateActiveColumnVect(activeCols, inhibCols, colOverlapVect)

#activeCol = calculateTfActiveCol(colOverlapMat)


# # Start training
# with tf.Session() as sess:
#     print("Run Sess")
#     tf_activeCol = activeCol.eval()
#     print(tf_activeCol)


# np_activeCol = calculateActiveCol(colOverlapMat)
# print("numpy activeCol = \n%s" % np_activeCol)
