
import numpy as np
from numpy import array, arange, ix_
from datetime import datetime
import random

import theano.tensor as T
from theano import function, shared
from theano.sandbox.neighbours import images2neibs
from theano import tensor
from theano.tensor import set_subtensor

from theano.tensor.sort import argsort, sort
from theano import Mode
import math

import cProfile

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
col_pat = T.matrix(dtype='int32')
act_cols = T.matrix(dtype='float32')
col_num2 = T.matrix(dtype='int32')
row_numMat4 = T.matrix(dtype='int32')
cur_inhib_cols4 = T.vector(dtype='int32')

test_meInhib = T.switch(T.eq(cur_inhib_cols4[row_numMat4], 1), 0, 1)
set_winners = act_cols[col_pat-1, col_num2]
check_colNotInhib = T.switch(T.lt(cur_inhib_cols4[col_pat-1], 1), set_winners, test_meInhib)
check_colNotPad = T.switch(T.ge(col_pat-1, 0), check_colNotInhib, 0)
get_activeColMat = function([act_cols,
                             col_pat,
                             col_num2,
                             row_numMat4,
                             cur_inhib_cols4],
                            check_colNotPad,
                            on_unused_input='warn',
                            allow_input_downcast=True
                            )


def calculateActiveColumnVect(activeCols, inhibCols, colOverlapVect):
    # Calculate for each column a list of columns which that column can
    # be inhibited by. Set the winning columns in this list as one.
    # If a column is inhibited already then all those positions in
    # the colwinners relating to that col are set as one. This means
    # the inhibited columns don't determine the active columns
    colwinners = get_activeColMat(activeCols,
                                  colInConvoleList,
                                  col_num,
                                  row_numMat,
                                  inhibCols)

    print("colwinners = \n%s" % colwinners)

    return colwinners

colInConvoleList = np.array(
                    [[5, 4, 2, 1],
                     [6, 5, 3, 2],
                     [0, 6, 0, 3],
                     [8, 7, 5, 4],
                     [9, 8, 6, 5],
                     [0, 9, 0, 6],
                     [0, 0, 8, 7],
                     [0, 0, 9, 8],
                     [0, 0, 0, 9]])
print("colInConvoleList = \n %s" % colInConvoleList)

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
activeCols = np.array(
                    [[0, 0, 0, 0],
                     [1, 1, 1, 0],
                     [0, 0, 0, 1],
                     [0, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 1, 0, 1],
                     [1, 0, 1, 0],
                     [1, 0, 0, 0],
                     [0, 0, 1, 1]])
print("activeCols = \n%s" % activeCols)
# Create just a vector storing if a column is inhibited or not
inhibCols = np.array([random.randint(0, 1) for i in range(width*height)])
inhibCols = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
print("inhibCols = \n%s" % inhibCols)
colOverlapMat = np.random.randint(10, size=(3, 3))
colOverlapMat = np.array(
                         [[9, 9, 9],
                          [0, 9, 0],
                          [0, 0, 0]])
print("colOverlapMat = \n%s" % colOverlapMat)

tieBreaker = ([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9]])

overlapsGridTie = colOverlapMat + tieBreaker
# Create a vector of the overlap values for each column
colOverlapVect = overlapsGridTie.flatten()


activeColVect = calculateActiveColumnVect(activeCols, inhibCols, colOverlapVect)
