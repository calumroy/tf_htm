import tensorflow as tf
import numpy as np
from numpy import array, arange, ix_

# # Create the theano function for calculating
# # the minOverlap from the sorted vector of overlaps for each column.
# # This function takes a vector of indicies indicating where the
# # minLocalActivity resides for each row in the matrix.
# # Note: the sorted overlap matrix goes from low to highest so use neg index.
# self.min_OIndex = T.vector(dtype='int32')
# self.s_ColOMat = T.matrix(dtype='float32')
# self.row_numVect2 = T.vector(dtype='int32')
# self.get_indPosVal = self.s_ColOMat[self.row_numVect2, -self.min_OIndex]
# self.get_minLocAct = function([self.min_OIndex,
#                                self.s_ColOMat,
#                                self.row_numVect2],
#                               self.get_indPosVal,
#                               allow_input_downcast=True
#                               )

# # Create the theano function for calculating
# # if a column should be active or not based on whether it
# # has an overlap greater then or equal to the minLocalActivity.
# self.minLocalActivity = T.matrix(dtype='float32')
# self.colOMat = T.matrix(dtype='float32')
# self.check_gt_zero = T.switch(T.gt(self.colOMat, 0), 1, 0)
# self.check_gteq_minLocAct = T.switch(T.ge(self.colOMat, self.minLocalActivity), self.check_gt_zero, 0)
# #self.indexActCol = tensor.eq(self.check_gteq_minLocAct, 1).nonzero()
# self.get_activeCol = function([self.colOMat,
#                               self.minLocalActivity],
#                               self.check_gteq_minLocAct,
#                               on_unused_input='warn',
#                               allow_input_downcast=True
#                               )


def sortOverlapMatrix(colOverlapVals):
    # colOverlapVals, each row is a list of overlaps values that
    # a column can potentially inhibit.
    # Sort the grid of overlap values from largest to
    # smallest for each columns inhibit overlap vect.
    # sortedColOverlapsVals = self.sort_vect(colOverlapVals, 1)
    sortedColOverlapsVals = np.sort(colOverlapVals, axis=1)

    print("sortedColOverlapsVals = \n%s" % sortedColOverlapsVals)
    return sortedColOverlapsVals


def get_minLocAct(minOverlapIndex, sortedColOverlapMat, row_numVect):
    print("minOverlapIndex = \n%s" % minOverlapIndex)
    print("sortedColOverlapMat = \n%s" % sortedColOverlapMat)
    print("row_numVect = \n%s" % row_numVect)

    minLocAct = sortedColOverlapMat[row_numVect, -minOverlapIndex]
    print("minLocAct = \n%s" % minLocAct)
    return minLocAct


def get_activeCol(colOverlapMat, minLocalAct):
    # np.greater(colOverlapMat, minLocalAct)
    outMat = (colOverlapMat > minLocalAct)
    outMat = array(outMat > 0, dtype=int)

    print("outMat = \n%s" % outMat)
    return outMat


def calculateActiveCol(colOverlapMat):
    # Calculate the active columns. The columns that have a higher overlap
    # value then the neighbouring ones.

    numCols = colOverlapMat.shape[0]
    print("numCols = ", numCols)

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

    print("minLocalAct = \n%s" % minLocalAct)

    # First take the colOverlaps matrix and flatten it into a vector.
    # Broadcast minLocalActivity so it is the same dim as colOverlapMat
    widthColOverlapMat = len(sortedColOverlapMat[0])
    minLocalAct = np.tile(np.array([minLocalAct]).transpose(), (1, widthColOverlapMat))

    print("minLocalAct = \n%s" % minLocalAct)
    print("colOverlapMat = \n%s" % colOverlapMat)

    # Now calculate for each columns list of overlap values, which ones are larger
    # then the minLocalActivity number.
    activeCols = get_activeCol(colOverlapMat, minLocalAct)

    print("sortedColOverlapMat = \n%s" % sortedColOverlapMat)
    print("minLocalAct = \n%s" % minLocalAct)
    print("colOverlapMat = \n%s" % colOverlapMat)
    print("activeCols = \n%s" % activeCols)

    return activeCols

def calculateTfActiveCol(colOverlapMat):
    # Calculate the active columns. The columns that have a higher overlap
    # value then the neighbouring ones.

    numCols = colOverlapMat.shape[0]
    print("numCols = ", numCols)

    desiredLocalActivity = 2
    minOverlapIndex = np.array([desiredLocalActivity for i in range(numCols)])

    # Create just a vector storing the row numbers for each column.
    # This is just an incrementing vector from zero to the number of columns - 1
    row_numVect = np.array([i for i in range(numCols)])

    # Sort the colOverlap matrix for each row. A row holds the inhib overlap
    # values for a single column.
    sortedColOverlapMat = tf.contrib.framework.sort(colOverlapMat,
                                                    axis=-1,
                                                    direction='ASCENDING',
                                                    name='sortedColOverlapMat')

    print("sortedColOverlapMat = \n%s" % sortedColOverlapMat)
    print("row_numVect = \n%s" % row_numVect)
    print("-minOverlapIndex = \n%s" % -minOverlapIndex)
    # Get the minLocalActivity for each col.
    #minLocalAct = sortedColOverlapMat[:row_numVect, -minOverlapIndex]
    minLocalAct = tf.gather(sortedColOverlapMat, -minOverlapIndex, axis=0)

    numCols, widthColOverlapMat = sortedColOverlapMat.shape
    minLocalAct = tf.tile(tf.transpose(minLocalAct), (1, widthColOverlapMat))

    #print("sortedColOverlapMat = \n%s" % sortedColOverlapMat)
    print("minLocalAct = \n%s" % minLocalAct)
    # print("colOverlapMat = \n%s" % colOverlapMat)
    # print("activeCols = \n%s" % activeCols)

    return activeCols


colOverlapMat = np.random.randint(10, size=(9, 9))
print("colOverlapMat = ", colOverlapMat)

activeCol = calculateTfActiveCol(colOverlapMat)

print("activeCol = ", activeCol)

# # Start training
# with tf.Session() as sess:


#     print(c2.eval())