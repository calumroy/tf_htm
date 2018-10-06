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
        self.numColumns = self.width * self.height
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
        # How far to move the kernal when working out a columns neighbours thar
        # it can inhibit.
        self.stepY = 1
        self.stepX = 1
        # Center the inhibition areas over the column that is inhibiting the others.
        self.centerPotSynapses = 1
        # Wrap the inhibition areas so columns can inhibit other columns on opposite edges.
        self.wrapInput = 1

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

        # Now Also calculate a convole grid so the columns position
        # in the resulting col inhib overlap matrix can be tracked.
        self.incrementingMat = np.array([[1+i+self.width*j for i in range(self.width)] for j in range(self.height)])

        ############################################
        # Create tensorflow variables and functions
        ############################################

        # Set the location to store the tensflow logs so we can run tensorboard.
        self.logsPath = '/tmp/tensorflow_logs/example/tf_htm/'
        now = datetime.now()
        self.logsPath = self.logsPath + now.strftime("%Y%m%d-%H%M%S") + "/"

        # Calculate the overlaps associated with columns that can be inhibited.
        self.colConvolePatternIndex = self.getColInhibInputs(self.incrementingMat)

        # Calculate a matrix storing the location of the numbers from
        # colConvolePatternIndex.
        self.colInConvoleList = self.calculateConvolePattern(self.colConvolePatternIndex)

        # Store a vector where each element stores for a column how many times
        # that column appears in other columns convole lists.
        #self.nonPaddingSumVect = self.get_gtZeroMat(self.colInConvoleList)
        #self.nonPaddingSumVect = self.get_sumRowMat(self.nonPaddingSumVect)

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

    def calculateConvolePattern(self, inputGrid):
        '''
        Determine the row number locations of the column
        numbers in the inputGrid.

        eg.
        inputGrid                   Calculated output
        [[  0.   0.   0.   1.]      [[  7.   6.   2.   1.]
         [  0.   0.   1.   2.]       [  0.   7.   3.   2.]
         [  0.   0.   2.   3.]       [  0.   0.   4.   3.]
         [  0.   0.   3.   4.]       [  0.   0.   5.   4.]
         [  0.   0.   4.   5.]       [  0.   0.   0.   5.]
         [  0.   1.   0.   6.]       [  0.   0.   7.   6.]
         [  1.   2.   6.   7.]]      [  0.   0.   0.   7.]]

         Note: height = numCols = self.width * self.height
        '''

        print "inputGrid = \n%s" % inputGrid
        width = len(inputGrid[0])
        height = len(inputGrid)

        rolledInputGrid = np.array([[0 for i in range(width)] for j in range(height)])
        outputGrid = np.array([[0 for i in range(width)] for j in range(height)])
        firstNonZeroIndVect = np.array([0 for i in range(width)])
        firstNonZeroVect = np.array([0 for i in range(width)])

        #print "width = %s height = %s" % (width, height)
        print "Setting Up tf inhibition calculator"
        for c in range(width):
            #print "c = %s" % c
            # Search for the column numbers.
            # They are always in order down the column
            # Now roll each column in the inputGrid upwards by the
            # this is a negative numpy roll.
            for r in range(height):
                #print "r = %s" % r
                firstNonZero = int(inputGrid[r, c])
                if firstNonZero > 0.0:
                    firstNonZeroIndVect[c] = r
                    firstNonZeroVect[c] = firstNonZero
                    rolledInputGrid[:, c] = np.roll(inputGrid[:, c], (-r+firstNonZero-1), axis=0)
                    break
        print "Done"

        #print "inputGrid = \n%s" % inputGrid

        #print "firstNonZeroIndVect = \n%s" % firstNonZeroIndVect
        #print "firstNonZeroVect = \n%s" % firstNonZeroVect

        with tf.name_scope('addToConvolePat'):
            # Create the tf graph for calculating
            # the colInConvole matrix. This takes a vector
            # storing an offset number and adds this to the input
            # matrix if the element in the input matrix is greater then
            # zero.
            # We have to cast the output of the comparison from a bool to a float so it can be multiplied.

            convolePat = tf.multiply(
                                     tf.cast(tf.greater(rolledInputGrid, 0), tf.float32),
                                     tf.cast(rolledInputGrid +
                                             tf.gather(firstNonZeroIndVect, self.col_num) -
                                             tf.gather(firstNonZeroVect, self.col_num) + 1,
                                             tf.float32)
                                    )

        # Run the tf graph
        with tf.Session() as sess:
            outputGrid = sess.run(convolePat)

        print("outputGrid = \n%s" % outputGrid)

        return outputGrid

    def getColInhibInputs(self, inputGrid):
        # Take the input and put it into a 4D tensor.
        # This is because the tensorflow function extract_image_patches
        # works with 4D tensors only.
        inputGrid = np.array([[inputGrid]])

        # print "inputGrid.shape = %s,%s,%s,%s" % inputGrid.shape
        firstDim, secondDim, width, height = inputGrid.shape

        # Calculate the input overlaps for each column.
        # Create the variables to use in the convolution.
        with tf.name_scope('getColInhibInputs'):
            inputInhibCols = self.poolInputs(inputGrid)

        # Run the tf graph
        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())

            inputInhibCols = sess.run(self.colConvole)

        print "getColInhibInputs Input = \n%s" % inputGrid
        print "inputInhibCols = \n%s" % inputInhibCols
        #print "inputInhibCols.shape = %s,%s" % inputInhibCols.shape
        return inputInhibCols

    def getPaddingSizes(self):
        # Calculate the amount of padding to add to each dimension.
        topPos_y = 0
        bottomPos_y = 0
        leftPos_x = 0
        rightPos_x = 0

        # This calculates how much of the input grid is not covered by
        # the htm grid in each dimension using the step sizes.
        leftOverWidth = self.width - (1 + (self.width - 1) * self.stepX)
        leftOverHeight = self.height - (1 + (self.height - 1) * self.stepY)

        if self.centerPotSynapses == 0:
            # The potential synapses are not centered over the input
            # This means only the right side and bottom of the input
            # need padding.
            topPos_y = 0
            bottomPos_y = int(math.floor(self.potentialHeight-1) - math.floor(leftOverHeight))
            leftPos_x = 0
            rightPos_x = int(math.floor(self.potentialWidth-1) - math.floor(leftOverWidth))

        else:
            # The potential synapses are centered over the input
            # This means all sides of the input may need padding
            topPos_y = int(math.ceil(float(self.potentialHeight-1)/2) - math.ceil(float(leftOverHeight)/2))
            bottomPos_y = int(math.floor(float(self.potentialHeight-1)/2) - math.floor(float(leftOverHeight)/2))

            leftPos_x = int(math.ceil(float(self.potentialWidth-1)/2) - math.ceil(float(leftOverWidth)/2))
            rightPos_x = int(math.floor(float(self.potentialWidth-1)/2) - math.floor(float(leftOverWidth)/2))

        # Make sure all are larger then zero still
        if topPos_y < 0:
            topPos_y = 0
        if bottomPos_y < 0:
            bottomPos_y = 0
        if leftPos_x < 0:
            leftPos_x = 0
        if rightPos_x < 0:
            rightPos_x = 0
        # Return the padding sizes.
        # The topPos_y is how many columns to add to pad above the first row of elements.
        # The bottomPos_y is how many columns to add to pad below the last row of elements.
        return [topPos_y, bottomPos_y, leftPos_x, rightPos_x]

    def poolInputs(self, inputGrid):
        # Create the variables to use in the convolution.
        with tf.name_scope('poolInputs'):
            # A variable to store the kernal used to workout the potential pool for each column
            # The kernal must be a 4d tensor.
            self.kernel = [1, self.potentialHeight, self.potentialWidth, 1]
            # The image must be a 4d tensor for the convole function. Add some extra dimensions.
            self.image = tf.reshape(inputGrid, [1, self.height, self.width, 1], name='image')
            # Create the stride for the convolution
            self.stride = [1, self.stepY, self.stepX, 1]

            # Create the padding sizes to use and the padding node.
            with tf.name_scope('padding'):
                [padU, padD, padL, padR] = self.getPaddingSizes()
                # print("[padL, padR, padU, padD] = %s, %s, %s, %s" % (padL, padR, padU, padD))
                self.paddings = tf.constant([[0, 0], [padU, padD], [padL, padR], [0, 0]])
                # set the padding
                if self.wrapInput is True:
                    # We will add our own padding so we can reflect the input.
                    # This prevents the edges from being unfairly hindered due to a smaller overlap with the kernel.
                    self.paddedInput = tf.pad(self.image, self.paddings, "REFLECT")
                else:
                    self.paddedInput = tf.pad(self.image, self.paddings, "CONSTANT")

            # From the padded input for each HTM column get a list of the potential inputs that column can connect to.
            # Rates can be used to set how many pixels between each slected pixel are skipped.
            self.imageNeib4d = tf.extract_image_patches(images=self.paddedInput,
                                                        ksizes=self.kernel,
                                                        strides=self.stride,
                                                        rates=[1, 1, 1, 1],
                                                        padding='VALID')

            # print("kernel = \n%s" % self.kernel)
            # print("stride = \n%s" % self.stride)
            # print("image = \n%s" % self.image)
            # print("paddedInput = \n%s" % self.paddedInput)
            # print("imageNeib4d = \n%s" % self.imageNeib4d)

            # Reshape rates as the output is 3 dimensional and we only require 2 dimensions
            # A list for each of the htm columns that shows the results of each columns convolution
            self.imageNeib = tf.reshape(self.imageNeib4d,
                                        [self.numColumns,
                                         self.potentialHeight*self.potentialWidth],
                                        name='overlapImage')

            # tf.squeeze Removes dimensions of size 1 from the shape of a tensor.
            self.colConvole = tf.squeeze(self.imageNeib, name="colConvole")



if __name__ == '__main__':

    potWidth = 3
    potHeight = 3
    centerInhib = 1
    numRows = 4
    numCols = 4
    desiredLocalActivity = 2
    minOverlap = 1

    # Some made up inputs to test with
    #colOverlapGrid = np.random.randint(1, size=(numRows, numCols))
    colOverlapGrid = np.array([[8, 4, 5, 8],
                               [8, 6, 1, 6],
                               [7, 7, 9, 4],
                               [2, 3, 1, 5]])
    print "colOverlapGrid = \n%s" % colOverlapGrid

    inhibCalculator = inhibitionCalculator(numCols, numRows,
                                           potWidth, potHeight,
                                           desiredLocalActivity,
                                           minOverlap,
                                           centerInhib)

    #cProfile.runctx('activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)', globals(), locals())
    #activeColumns = inhibCalculator.calculateWinningCols(colOverlapGrid)

    #activeColumns = activeColumns.reshape((numRows, numCols))
    #print "colOverlapGrid = \n%s" % colOverlapGrid
    #print "activeColumns = \n%s" % activeColumns







