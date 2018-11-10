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


    # numActColsInConVect = self.get_sumRowMat(actColsInCon)
    #     # print "numActColsInConVect = \n%s" % numActColsInConVect
    #     inhibitedColsConMat = self.check_actColsCon(self.desiredLocalActivity,
    #                                                 self.colConvolePatternIndex,
    #                                                 numActColsInConVect,
    #                                                 actColsInCon
    #                                                 )


    return actColsInCon


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

potentialWidth = 2
potentialHeight = 2
width = 3
height = 3


#print("colOverlapMat = \n%s" % colOverlapMat)

activeColVect = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
print("activeColVect = \n%s" % activeColVect)

actColsInCon = calculateInhibCols(activeColVect, colConvolePatternIndex)

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
