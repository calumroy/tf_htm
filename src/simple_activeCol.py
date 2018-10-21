import tensorflow as tf
import numpy as np
from numpy import array, arange, ix_

# Create a tensorflow and numpy functions for calculating
# the active columns given their overlap values and the desired
# local activity. The columns that have a higher overlap
# value then the neighbouring ones.


def sortOverlapMatrix(colOverlapVals):
    # colOverlapVals, each row is a list of overlaps values that
    # a column can potentially inhibit.
    # Sort the grid of overlap values from largest to
    # smallest for each columns inhibit overlap vect.
    # sortedColOverlapsVals = self.sort_vect(colOverlapVals, 1)
    sortedColOverlapsVals = np.sort(colOverlapVals, axis=1)

    # print("sortedColOverlapsVals = \n%s" % sortedColOverlapsVals)
    return sortedColOverlapsVals


def get_minLocAct(minOverlapIndex, sortedColOverlapMat, row_numVect):
    # print("minOverlapIndex = \n%s" % minOverlapIndex)
    # print("sortedColOverlapMat = \n%s" % sortedColOverlapMat)
    # print("row_numVect = \n%s" % row_numVect)

    minLocAct = sortedColOverlapMat[row_numVect, -minOverlapIndex]
    # print("minLocAct = \n%s" % minLocAct)
    return minLocAct


def get_activeCol(colOverlapMat, minLocalAct):
    # np.greater(colOverlapMat, minLocalAct)
    outMat = (colOverlapMat > minLocalAct)
    outMat = array(outMat > 0, dtype=int)

    # print("outMat = \n%s" % outMat)
    return outMat


def calculateActiveCol(colOverlapMat):
    # Calculate the active columns. The columns that have a higher overlap
    # value then the neighbouring ones.

    numCols = colOverlapMat.shape[0]
    # print("numCols = ", numCols)

    desiredLocalActivity = 2
    minOverlapIndex = np.array([desiredLocalActivity for i in range(numCols)])

    # Create just a vector storing the row numbers for each column.
    # This is just an incrementing vector from zero to the number of columns - 1
    row_numVect = np.array([i for i in range(numCols)])

    # Sort the colOverlap matrix for each row. A row holds the inhib overlap
    # values for a single column.
    sortedColOverlapMat = sortOverlapMatrix(colOverlapMat)
    # Get the minLocalActivity for each col.
    minLocalAct = get_minLocAct(minOverlapIndex,
                                sortedColOverlapMat,
                                row_numVect)

    # print("minLocalAct = \n%s" % minLocalAct)

    # First take the colOverlaps matrix and flatten it into a vector.
    # Broadcast minLocalActivity so it is the same dim as colOverlapMat
    widthColOverlapMat = len(sortedColOverlapMat[0])
    minLocalAct = np.tile(np.array([minLocalAct]).transpose(), (1, widthColOverlapMat))

    print("minLocalAct.shape = \n", minLocalAct.shape)
    print("minLocalAct = \n%s" % minLocalAct)
    # print("colOverlapMat = \n%s" % colOverlapMat)

    # Now calculate for each columns list of overlap values, which ones are larger
    # then the minLocalActivity number.
    activeCols = get_activeCol(colOverlapMat, minLocalAct)

    # print("sortedColOverlapMat = \n%s" % sortedColOverlapMat)
    # print("minLocalAct = \n%s" % minLocalAct)
    # print("colOverlapMat = \n%s" % colOverlapMat)
    # print("activeCols = \n%s" % activeCols)

    return activeCols


def calculateTfActiveCol(colOverlapMat):
    # Calculate the active columns. The columns that have a higher overlap
    # value then the neighbouring ones.

    numCols, numInhib = colOverlapMat.shape
    print("numCols = ", numCols)
    print("numInhib = ", numInhib)

    desiredLocalActivity = 2
    minOverlapIndex = np.array([desiredLocalActivity for i in range(numCols)])

    # Create just a vector storing the row numbers for each column.
    # This is just an incrementing vector from zero to the number of columns - 1
    row_numVect = np.array([i for i in range(numCols)])
    rowNumVect = tf.constant(row_numVect, dtype=tf.int32)

    # Sort the colOverlap matrix for each row. A row holds the inhib overlap
    # values for a single column.
    sortedColOverlapMat = tf.contrib.framework.sort(colOverlapMat,
                                                    axis=1,
                                                    direction='ASCENDING',
                                                    name='sortedColOverlapMat')

    # print("sortedColOverlapMat = \n%s" % sortedColOverlapMat)

    # Get the minLocalActivity for each col.
    colIndicies = tf.subtract(numInhib, minOverlapIndex)
    print("rowNumVect = \n%s" % rowNumVect)
    print("colIndicies = \n%s" % colIndicies)

    indicies = tf.stack([rowNumVect, colIndicies], axis=-1)
    minLocalAct = tf.gather_nd(sortedColOverlapMat, indicies)

    # numCols, widthColOverlapMat = sortedColOverlapMat.shape
    # minLocalAct = tf.tile(tf.transpose(minLocalAct), (1, widthColOverlapMat))

    # print("sortedColOverlapMat = \n%s" % sortedColOverlapMat)
    # print("minLocalAct.shape = \n%s" % minLocalAct.shape)
    print("minLocalAct = \n%s" % minLocalAct)

    # Now calculate for each columns list of overlap values, which ones are larger
    # then the minLocalActivity number.
    # Broadcast the minLocalAct so it has the same dim as colOverlapMat
    # This is acheived by adding an extra empty dimension to the minLocalAct then broadcasting
    expMinLocalAct = tf.expand_dims(minLocalAct, 1)
    bdim = [numCols, numInhib]
    print("bdim = " , bdim)
    minLocalActExpand = tf.broadcast_to(expMinLocalAct, bdim)
    activeColsTemp = tf.cast(tf.greater(colOverlapMat, minLocalActExpand),
                             tf.float32)
    activeCols = tf.cast(tf.greater(activeColsTemp, 0),
                         tf.float32)

    # Print the output.
    with tf.name_scope('print2'):
        # Use a print node in the graph. The first input is the input data to pass to this node,
        # the second is an array of which nodes in the graph you would like to print
        # pr_mult = tf.Print(mult_y, [mult_y, newGrid], summarize = 25)
        pr_mult = tf.Print(activeCols,
                           [minLocalActExpand, indicies, sortedColOverlapMat],
                           message="Print", summarize=200)

    print("activeCols = \n%s" % activeCols)

    return pr_mult


colOverlapMat = np.random.randint(10, size=(9, 4))
print("colOverlapMat = \n%s" % colOverlapMat)

activeCol = calculateTfActiveCol(colOverlapMat)


# Start training
with tf.Session() as sess:
    print("Run Sess")
    tf_activeCol = activeCol.eval()
    print(tf_activeCol)


np_activeCol = calculateActiveCol(colOverlapMat)
print("numpy activeCol = \n%s" % np_activeCol)

# Mkae sure the numpy and tensorflow implementations are the same.
assert(np_activeCol.all() == tf_activeCol.all())

