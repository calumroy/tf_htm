from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
from numpy import array, arange, ix_
from datetime import datetime
import random


def calculateInhibCols(activeColumnVect, colOverlapVect):
    # Now calculate a list of inhibited columns.
    # Create a vector one element for each col. 1 means the col has
    # been inhibited.

    # Calculates which columns convole group contains columns that
    # have too many active columns in their convole groups.
    # Do not include columns that are active.
    # If a column A finds that it contains active column B then check the
    # convole group of B. If B contains in its convole group the number
    # of desired local activities then the original column A should be inhibited.

    # Calculate which columns in a columns convole inhib list are active.
    # A Matrix is returned where each row stores a list of
    # ones or zeros indicating which columns in a columns convole
    # group are active.
    # This is what the actColsInCon tensor represents.
    with tf.name_scope('get_actColsInCon'):
        actColsVect = tf.convert_to_tensor(activeColumnVect, dtype=tf.int32, name='activeColumnVect')
        colConvolePatInd = tf.convert_to_tensor(colConvolePatternIndex, dtype=tf.int32, name='colConvolePatternIndex')

        zeros_mat = tf.zeros_like(colConvolePatInd, dtype=tf.int32, name='zeros_mat')

        col_convolePatInd_negone = tf.add(colConvolePatInd, -1)
        check_rCols = tf.gather(actColsVect, tf.maximum(col_convolePatInd_negone, zeros_mat), name='check_rCols')

        actColsInCon = tf.where(tf.greater(colConvolePatInd, 0), check_rCols, zeros_mat)
        numActColsInConVect = tf.reduce_sum(actColsInCon, 1)

    # Calculate whether the active column in the columns convole list contains
    # the desired local activity number of active columns in its
    # convole list. This function returns a matrix where each
    # row stores a list of ones or zeros indicating which
    # columns in a columns convole group contain the desired number
    # of active columns.
    with tf.name_scope('check_actColsCon'):
        ones_mat = tf.ones_like(colConvolePatInd, dtype=tf.int32, name='zeros_mat')
        get_colsInVect = tf.gather(numActColsInConVect, tf.maximum(col_convolePatInd_negone, zeros_mat), name='check_rCols')

        get_colsConPat = tf.where(tf.greater_equal(get_colsInVect, desiredLocalActivity), ones_mat, zeros_mat)
        check_colsConPat = tf.where(tf.greater(actColsInCon, 0), get_colsConPat, zeros_mat)

        inhibitedColsConMat = check_colsConPat

    # Calculate the sum of the rows for the non active columns.
    # An input matrix elements represent the columns convole
    # lists where each column in the list is one if that columns
    # convole list also contains the desired local activity number of
    # active columns. The other input vector is a list of the active columns.
    with tf.name_scope('sum_nonActColsRows'):
        zeros_vec = tf.zeros_like(activeColumnVect, dtype=tf.int32, name='zeros_vec')
        get_rowSum = tf.reduce_sum(inhibitedColsConMat, 1)
        check_colAct = tf.where(tf.greater(actColsVect, 0), zeros_vec, get_rowSum)

        inhibitedColsVect = check_colAct
    # A column should also be inhibited if it is in an active columns convole
    # group and that active column already contains the desired local activity
    # number of active columns.
    # If column A is active, it contains column B in its convole group and
    # it already contains the desired local activity number of active columns
    # in its convole group then inhibit column B. This check must be done because
    # sometimes sizes of the potential width and height mean that column B
    # can be in columns A convole group but column A is not in columns B's.

    # For the active columns if their convole list contains
    # the desired local activity number of active columns then
    # inhibit the remaining unactive cols in the convole list.
    # This function returns a matrix where each
    # row stores a list of ones or zeros indicating which
    # columns in a columns convole group will be inhibited.
    with tf.name_scope('inhibit_actColsCon'):
        colInConvoleListTf = tf.convert_to_tensor(colInConvoleList, dtype=tf.int32, name='colInConvoleListTf')
        colInConvoleList_negone = tf.add(colInConvoleListTf, -1)
        colin_numActColsInConVect = tf.gather(numActColsInConVect, tf.maximum(colInConvoleList_negone, zeros_mat))

        check_numActCols = tf.where(tf.greater_equal(colin_numActColsInConVect, desiredLocalActivity), ones_mat, zeros_mat)
        colin_actColVect = tf.gather(actColsVect, tf.maximum(colInConvoleList_negone, zeros_mat))
        # Elementwise And applied to colin_actColVect and check_numActCols
        check_colIndAct = tf.math.multiply(colin_actColVect, check_numActCols)

        get_rowActiveColumnVect = tf.gather(actColsVect, row_numMat)
        check_colsRowInAct = tf.where(tf.greater(get_rowActiveColumnVect, 0), zeros_mat, check_colIndAct)
        check_gtZero = tf.where(tf.greater(colInConvoleListTf, 0), check_colsRowInAct, zeros_mat)

        inhibitedColsConMat2 = check_gtZero

    # Now also calculate which columns convole groups contain too
    # many active cols and should therfore be inhibited.
    # If the column is active do not include it.
    with tf.name_scope('inhibited_ColsVect'):
        ones_vec = tf.ones_like(activeColumnVect, dtype=tf.int32, name='ones_vec')
        inhibitedColsVect2 = tf.reduce_sum(inhibitedColsConMat2, 1)
        # Calculate the input columns vector where the active columns
        # in the input vector have been set to zero.
        numActColsInConVect3 = tf.where(tf.greater(actColsVect, 0), zeros_vec, numActColsInConVect)
        # Calculate if an input vector is larger then a scalar (element wise) desiredLocalActivity.
        inhibitedColsVect3 = tf.where(tf.greater(numActColsInConVect3, desiredLocalActivity), ones_vec, zeros_vec)
        # All three inhibitedColsVect vectors indicate that the column should be inhibited.
        # Add them together to find the total inhibited columns.
        inhibColsVector1 = tf.add(inhibitedColsVect, inhibitedColsVect2)
        inhibColsVector2 = tf.add(inhibColsVector1, inhibitedColsVect3)
        # Now see which columns appeared in either list of inhibited columns
        gt_zeroVect = tf.cast(tf.greater(inhibColsVector2, 0), tf.int32)
        # If the column has a zero overlap value (ie its overlap value
        # plus the tiebreaker is less then one then inhibit the column.
        colOverVect = tf.convert_to_tensor(colOverlapVect, dtype=tf.float32, name='colOverlapVect')
        inhibCols = tf.where(tf.less(colOverVect, 1.0), ones_vec, gt_zeroVect)

    with tf.name_scope('get_notInhibOrActNum'):
        # Sum the InhibCols vector and compare to the number of cols
        # If equal then all columns have been inhibited or are active.
        tf_width = tf.constant(width, dtype=tf.int32, name='width')
        tf_height = tf.constant(height, dtype=tf.int32, name='height')
        red_inhibCols = tf.reduce_sum(inhibCols, 0)
        red_actColVect = tf.reduce_sum(actColsVect, 0)
        print(red_inhibCols)
        print(red_actColVect)
        notInhibOrActNum = tf_width * tf_height - red_inhibCols - red_actColVect

    # Print the output.
    with tf.name_scope('print'):
        # Use a print node in the graph. The first input is the input data to pass to this node,
        # the second is an array of which nodes in the graph you would like to print
        # pr_mult = tf.Print(mult_y, [mult_y, newGrid], summarize = 25)
        print_out = tf.print("\n colInConvoleList_negone = \n", colInConvoleList_negone,
                             "\n actColsVect = ", actColsVect,
                             "\n actColsInCon = \n", actColsInCon,
                             "\n numActColsInConVect = ", numActColsInConVect,
                             "\n colin_numActColsInConVect = \n", colin_numActColsInConVect,
                             "\n check_numActCols = \n", check_numActCols,
                             "\n colin_actColVect = \n", colin_actColVect,
                             "\n check_colIndAct = \n", check_colIndAct,
                             "\n get_rowActiveColumnVect = \n", get_rowActiveColumnVect,
                             "\n check_colsRowInAct = \n", check_colsRowInAct,
                             "\n",
                             "\n inhibitedColsConMat2 = \n", inhibitedColsConMat2,
                             "\n inhibitedColsVect = ", inhibitedColsVect,
                             "\n inhibColsVector1 = ", inhibColsVector1,
                             "\n inhibitedColsVect2 = ", inhibitedColsVect2,
                             "\n numActColsInConVect = ", numActColsInConVect,
                             "\n numActColsInConVect3 = ", numActColsInConVect3,
                             "\n inhibitedColsVect3 = ", inhibitedColsVect3,
                             "\n inhibColsVector2 = ", inhibColsVector2,
                             "\n gt_zeroVect = ", gt_zeroVect,
                             "\n colOverVect = \n", colOverVect,
                             "\n inhibCols = ", inhibCols,
                             "\n actColsVect = ", actColsVect,
                             "\n notInhibOrActNum = ", notInhibOrActNum,
                             "\n",
                             output_stream=sys.stdout,
                             summarize=200)
        # print_out = tf.Print(notInhibOrActNum,
        #                      [inhibCols, red_inhibCols, red_actColVect,
        #                       colOverlapVect,
        #                       inhibColsVector2
        #                       ],
        #                      message="Print", summarize=200)


    return inhibCols, print_out


# colConvolePatternIndex = (
#                     [[0, 0, 0, 1],
#                      [0, 0, 1, 2],
#                      [0, 0, 2, 3],
#                      [0, 1, 0, 4],
#                      [1, 2, 4, 5],
#                      [2, 3, 5, 6],
#                      [0, 4, 0, 7],
#                      [4, 5, 7, 8],
#                      [5, 6, 8, 9]])

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

colConvolePatternIndex =  array([[ 0,  0,  0,  0,  1,  2,  0,  5,  6],
                                 [ 0,  0,  0,  1,  2,  3,  5,  6,  7],
                                 [ 0,  0,  0,  2,  3,  4,  6,  7,  8],
                                 [ 0,  0,  0,  3,  4,  0,  7,  8,  0],
                                 [ 0,  1,  2,  0,  5,  6,  0,  9, 10],
                                 [ 1,  2,  3,  5,  6,  7,  9, 10, 11],
                                 [ 2,  3,  4,  6,  7,  8, 10, 11, 12],
                                 [ 3,  4,  0,  7,  8,  0, 11, 12,  0],
                                 [ 0,  5,  6,  0,  9, 10,  0, 13, 14],
                                 [ 5,  6,  7,  9, 10, 11, 13, 14, 15],
                                 [ 6,  7,  8, 10, 11, 12, 14, 15, 16],
                                 [ 7,  8,  0, 11, 12,  0, 15, 16,  0],
                                 [ 0,  9, 10,  0, 13, 14,  0,  0,  0],
                                 [ 9, 10, 11, 13, 14, 15,  0,  0,  0],
                                 [10, 11, 12, 14, 15, 16,  0,  0,  0],
                                 [11, 12,  0, 15, 16,  0,  0,  0,  0]])

colInConvoleList =  array([[ 6,  5,  0,  2,  1,  0,  0,  0,  0],
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
desiredLocalActivity = 2

# Create a matrix that just holds the row number for each element
row_numMat = np.array([[j for i in range(potentialWidth*potentialHeight)]
                      for j in range(width*height)])

print("row_numMat = \n%s" % row_numMat)

#activeColVect = np.array([0, 0, 1, 0, 1, 0, 0, 1, 1])
activeColVect =  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])


print("activeColVect = \n%s" % activeColVect)

# colOverlapMat = np.array(
#                          [[8, 8, 8],
#                           [0, 9, 0],
#                           [0, 9, 8]])
colOverlapMat = np.array(
                [[8, 4, 5, 8],
                 [8, 6, 1, 6],
                 [7, 7, 9, 4],
                 [2, 3, 1, 5]])


print("colOverlapMat = \n%s" % colOverlapMat)

# tieBreaker = ([[0.1, 0.2, 0.3],
#               [0.4, 0.5, 0.6],
#               [0.7, 0.8, 0.9]])

tieBreaker = np.array(
                    [[0.05882353, 0.11764706, 0.17647059, 0.23529412],
                     [0.29411765, 0.35294118, 0.41176471, 0.47058824],
                     [0.52941176, 0.58823529, 0.64705882, 0.70588235],
                     [0.76470588, 0.82352941, 0.88235294, 0.94117647]])

overlapsGridTie = colOverlapMat + tieBreaker
# Create a vector of the overlap values for each column
colOverlapVect = overlapsGridTie.flatten()

actColsInCon, notInhibOrActNum = calculateInhibCols(activeColVect, colOverlapVect)

logsPath = '/tmp/tensorflow_logs/example/tf_htm/'
now = datetime.now()
logsPath = logsPath + now.strftime("%Y%m%d-%H%M%S") + "/"
summary_writer = tf.summary.FileWriter(logsPath, graph=tf.get_default_graph())

# Start training
with tf.Session() as sess:
    print("Run Sess")
    tf_activeColumns, notInhibOrActNum = sess.run([actColsInCon, notInhibOrActNum])

    print("tf_activeColumns = %s" % tf_activeColumns)
    print("notInhibOrActNum = %s" % notInhibOrActNum)


# np_activeCol = calculateActiveCol(colOverlapMat)
# print("numpy activeCol = \n%s" % np_activeCol)
