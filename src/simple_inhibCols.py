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
    # columns in a columns convole group should be inhibited.
    with tf.name_scope('inhibit_actColsCon'):
        colInConvoleList_negone = tf.add(colInConvoleList, -1)
        colin_numActColsInConVect = tf.gather(numActColsInConVect, tf.maximum(colInConvoleList_negone, zeros_mat))

        check_numActCols = tf.where(tf.greater_equal(colin_numActColsInConVect, desiredLocalActivity), ones_mat, zeros_mat)
        colin_actColVect = tf.gather(actColsVect, tf.maximum(colInConvoleList_negone, zeros_mat))
        check_colIndAct = tf.where(tf.greater(colin_actColVect, 0), check_numActCols, zeros_mat)

        get_rowActiveColumnVect = tf.gather(actColsVect, row_numMat)
        check_colsRowInAct = tf.where(tf.greater(get_rowActiveColumnVect, 0), zeros_mat, check_colIndAct)
        check_gtZero = tf.where(tf.greater(colInConvoleList, 0), check_colsRowInAct, zeros_mat)

        inhibitedColsConMat2 = check_gtZero

    # Now also calculate which columns convole groups contain too
    # many active cols and should therfore be inhibited.
    # If the column is active do not include it.
    with tf.name_scope('inhibited_ColsVect'):
        ones_vec = tf.zeros_like(activeColumnVect, dtype=tf.int32, name='ones_vec')
        inhibitedColsVect2 = tf.reduce_sum(inhibitedColsConMat2, 1)
        # Calculate the input columns vector where the active columns
        # in the input vector have been set to zero.
        numActColsInConVect3 = tf.where(tf.greater(actColsVect, 0), zeros_vec, numActColsInConVect)
        # Calculate if an input vector is larger then a scalar (element wise).
        inhibitedColsVect3 = tf.where(tf.greater(numActColsInConVect3, desiredLocalActivity), ones_vec, zeros_vec)
        inhibColsVector1 = tf.add(inhibitedColsVect, inhibitedColsVect2)
        inhibColsVector2 = tf.add(inhibColsVector1, inhibitedColsVect3)
        # Now see which columns appeared in either list of inhibited columns
        gt_zeroVect = tf.cast(tf.greater(inhibColsVector2, 0), tf.int32)
        # If the column has a zero overlap value (ie its overlap value
        # plus the tiebreaker is less then one then inhibit the column.
        inhibCols = tf.where(tf.less(colOverlapVect, 1), ones_vec, gt_zeroVect)

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
        print_out = tf.Print(notInhibOrActNum,
                             [inhibCols, red_inhibCols, red_actColVect],
                             message="Print", summarize=200)

    return print_out


colConvolePatternIndex = (
                    [[0, 0, 0, 1],
                     [0, 0, 1, 2],
                     [0, 0, 2, 3],
                     [0, 1, 0, 4],
                     [1, 2, 4, 5],
                     [2, 3, 5, 6],
                     [0, 4, 0, 7],
                     [4, 5, 7, 8],
                     [5, 6, 8, 9]])

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
desiredLocalActivity = 2

# Create a matrix that just holds the row number for each element
row_numMat = np.array([[j for i in range(potentialWidth*potentialHeight)]
                      for j in range(width*height)])

print("row_numMat = \n%s" % row_numMat)

activeColVect = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1])
print("activeColVect = \n%s" % activeColVect)

colOverlapMat = np.array(
                         [[8, 8, 8],
                          [0, 9, 0],
                          [0, 9, 8]])
print("colOverlapMat = \n%s" % colOverlapMat)

tieBreaker = ([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9]])

overlapsGridTie = colOverlapMat + tieBreaker
# Create a vector of the overlap values for each column
colOverlapVect = overlapsGridTie.flatten()

actColsInCon = calculateInhibCols(activeColVect, colOverlapVect)

logsPath = '/tmp/tensorflow_logs/example/tf_htm/'
now = datetime.now()
logsPath = logsPath + now.strftime("%Y%m%d-%H%M%S") + "/"
summary_writer = tf.summary.FileWriter(logsPath, graph=tf.get_default_graph())



# Start training
with tf.Session() as sess:
    print("Run Sess")
    tf_actColsInCon = actColsInCon.eval()
    print(tf_actColsInCon)


# np_activeCol = calculateActiveCol(colOverlapMat)
# print("numpy activeCol = \n%s" % np_activeCol)
