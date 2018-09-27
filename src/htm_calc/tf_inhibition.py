import tensorflow as tf
import numpy as np
import math
from datetime import datetime


'''
A class to calculate the inhibition of columns for a HTM layer.
This class uses tensorflow functions to speed up the computation.

Inputs:
It uses the overlap values for each column, expressed in matrix form.
It must be in a matrix so convolution can be used to determine
column neighbours.


Outputs:
It outputs a binary vector where each position indicates if a column
is active or not.

THIS IS A REINIMPLEMENTATION OF THE OLD INHIBITON CODE BELOW

    #print "length active columns before deleting = %s" % len(self.activeColumns)
    self.activeColumns = np.array([], dtype=object)
    #print "actve cols before %s" %self.activeColumns
    allColumns = self.columns.flatten().tolist()
    # Get all the columns in a 1D array then sort them based on their overlap value.
    #allColumns = allColumns[np.lexsort(allColumns.overlap, axis=None)]
    allColumns.sort(key=lambda x: x.overlap, reverse=True)
    # Now start from the columns with the highest overlap and inhibit
    # columns with smaller overlaps.
    for c in allColumns:
        if c.overlap > 0:
            # Get the neighbours of the column
            neighbourCols = self.neighbours(c)
            minLocalActivity = self.kthScore(neighbourCols, self.desiredLocalActivity)
            #print "current column = (%s, %s) overlap = %d min = %d" % (c.pos_x, c.pos_y,
            #                                                            c.overlap, minLocalActivity)
            if c.overlap > minLocalActivity:
                self.activeColumns = np.append(self.activeColumns, c)
                self.columnActiveAdd(c, timeStep)
                # print "ACTIVE COLUMN x,y = %s, %s overlap = %d min = %d" % (c.pos_x, c.pos_y,
                #                                                             c.overlap, minLocalActivity)
            elif c.overlap == minLocalActivity:
                # Check the neighbours and see how many have an overlap
                # larger then the minLocalctivity or are already active.
                # These columns will be set active.
                numActiveNeighbours = 0
                for d in neighbourCols:
                    if (d.overlap > minLocalActivity or self.columnActiveState(d, self.timeStep) is True):
                        numActiveNeighbours += 1
                # if less then the desired local activity have been set
                # or will be set as active then activate this column as well.
                if numActiveNeighbours < self.desiredLocalActivity:
                    #print "Activated column x,y = %s, %s numActiveNeighbours = %s" % (c.pos_x, c.pos_y, numActiveNeighbours)
                    self.activeColumns = np.append(self.activeColumns, c)
                    self.columnActiveAdd(c, timeStep)
                else:
                    # Set the overlap score for the losing columns to zero
                    c.overlap = 0
            else:
                # Set the overlap score for the losing columns to zero
                c.overlap = 0
        self.updateActiveDutyCycle(c)
        # Update the active duty cycle variable of every column
'''


class inhibitionCalculator():
    def __init__(self, width, height, potentialInhibWidth, potentialInhibHeight,
                 desiredLocalActivity, minOverlap, centerInhib=1,
                 colOverlapsGrid=None, potColOverlapsGrid=None):
        # Temporal Parameters
        ###########################################
        # Specifies if the potential synapses are centered
        # over the columns
        self.centerInhib = centerInhib
        self.width = width
        self.height = height
        self.potentialWidth = potentialInhibWidth
        self.potentialHeight = potentialInhibHeight
        self.areaKernel = self.potentialWidth * self.potentialHeight
        self.desiredLocalActivity = desiredLocalActivity
        self.minOverlap = minOverlap
        # Store how much padding is added to the input grid
        self.topPos_y = 0
        self.bottomPos_y = 0
        self.leftPos_x = 0
        self.rightPos_x = 0

        ################################################################
        # The folowing variables are used for indicies when looking up values
        # in matricies from within a tensorflow operation.
        ################################################################
        # Create a matrix that just holds the column number for each element
        self.col_num = np.array([[i for i in range(self.potentialWidth*self.potentialHeight)]
                                for j in range(self.width*self.height)])

        # Create a matrix that just holds the row number for each element
        self.row_numMat = np.array([[j for i in range(self.potentialWidth*self.potentialHeight)]
                                   for j in range(self.width*self.height)])

        # Create just a vector storing the row numbers for each column.
        # This is just an incrementing vector from zero to the number of columns - 1
        self.row_numVect = np.array([i for i in range(self.width*self.height)])

        # Create just a vector stroing if a column is inhibited or not
        self.inhibCols = np.array([0 for i in range(self.width*self.height)])

        # Create a vector of minOverlap indicies. This stores the position
        # for each col where the minOverlap resides, in the sorted Convole overlap mat
        self.minOverlapIndex = np.array([self.desiredLocalActivity for i in range(self.width*self.height)])

        # Now Also calcualte a convole grid so the columns position
        # in the resulting col inhib overlap matrix can be tracked.
        self.incrementingMat = np.array([[1+i+self.width*j for i in range(self.width)] for j in range(self.height)])

        #self.colConvolePatternIndex = self.getColInhibInputs(self.incrementingMat)

        # Calculate a matrix storing the location of the numbers from
        # colConvolePatternIndex.
        #self.colInConvoleList = self.calculateConvolePattern(self.colConvolePatternIndex)

        # Store a vector where each element stores for a column how many times
        # that column appears in other columns convole lists.
        #self.nonPaddingSumVect = self.get_gtZeroMat(self.colInConvoleList)
        #self.nonPaddingSumVect = self.get_sumRowMat(self.nonPaddingSumVect)

        ############################################
        # Create tensorflow variables and functions
        ############################################

        # Set the location to store the tensflow logs so we can run tensorboard.
        self.logsPath = '/tmp/tensorflow_logs/example/tf_htm/'
        now = datetime.now()
        self.logsPath = self.logsPath + now.strftime("%Y%m%d-%H%M%S") + "/"

        with tf.name_scope('tf_inhibition'):
            with tf.name_scope('inhibInputs'):
                # Check to see if the input tensors where given as a parameter or if we need to create our own placeholders.
                if colOverlapsGrid is None:
                    # A variable to hold a tensor storing the overlap values for each column
                    self.colOverlapsGrid = tf.placeholder(tf.float32, [self.height, self.width], name='colOverlapsGrid')
                else:
                    self.colOverlapsGrid = colOverlapsGrid
                if potColOverlapsGrid is None:
                    # Setup a matrix where each position represents a columns potential overlap.
                    self.potColOverlapsGrid = tf.placeholder(tf.float32, [self.height, self.width], name='colOverlapsGrid')
                else:
                    self.potColOverlapsGrid = potColOverlapsGrid

            self.mult_y = tf.multiply(self.potColOverlapsGrid, self.colOverlapsGrid)

            self.summary_writer = tf.summary.FileWriter(self.logsPath, graph=tf.get_default_graph())




