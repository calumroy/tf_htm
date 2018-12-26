from __future__ import print_function
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

        # Create a vector of minOverlap indicies. This stores the position
        # for each col where the minOverlap resides, in the sorted Convole overlap mat
        self.minOverlapIndex = np.array([self.desiredLocalActivity for i in range(self.width*self.height)])

        # Now Also calculate a convole grid so the columns position
        # in the resulting col inhib overlap matrix can be tracked.
        self.incrementingMat = np.array([[1+i+self.width*j for i in range(self.width)] for j in range(self.height)])

        # Take the colOverlapMat and add a small number to each overlap
        # value based on that row and col number. This helps when deciding
        # how to break ties in the inhibition stage. Note this is not a random value!
        # Make sure the tiebreaker contains values less then 1.
        normValue = 1.0/float(self.numColumns+1)
        self.tieBreaker = np.array([[(1+i+j*self.width)*normValue for i in range(self.width)] for j in range(self.height)])
        print("self.tieBreaker = \n%s" % self.tieBreaker)

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
        self.nonPaddingSumVect = self.calculateNonPaddingSumVect(self.colInConvoleList)

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

            with tf.name_scope('WinningCols'):
                # Create just a vector storing if a column is inhibited or not
                self.inhibColsZeros = tf.zeros(self.width*self.height, dtype=tf.int32, name='inhibColsZeros')

                with tf.name_scope('addTieBreaker'):
                    # Add the tieBreaker value to the overlap values.
                    self.overlapsGridTie = tf.add(self.colOverlapsGrid, self.tieBreaker)
                # Calculate the overlaps associated with columns that can be inhibited.
                with tf.name_scope('getColOverlapMatOrig'):
                    self.colOverlapMatOrig = self.poolInputs(self.overlapsGridTie)

                with tf.name_scope('getActiveColumnVect'):
                    # Create a vector of the overlap values for each column
                    self.colOverlapVect = tf.reshape(self.overlapsGridTie, [self.numColumns])

                    self.activeCols = self.calculateActiveCol(self.colOverlapMatOrig)
                    self.activeColumnVect = self.calculateActiveColumnVect(self.activeCols,
                                                                           self.inhibColsZeros,
                                                                           self.colOverlapVect)

                with tf.name_scope('getInhibColsVect'):
                    self.inhibCols, self.notInhibOrActNum = self.calculateInhibCols(self.activeColumnVect, self.colOverlapVect)


                # Run the while loop which keeps calucalting until notInhibOrActNumFinal = 0
                with tf.name_scope('whileLoop'):
                    (self.inhibColsFinal,
                        self.notInhibOrActNumFinal,
                        self.activeColumnVectFinal) = tf.while_loop(self.condition, self.body,
                                                                    [self.inhibCols,
                                                                     self.notInhibOrActNum,
                                                                     self.activeColumnVect],
                                                                    back_prop=False)
                # Print the output.
                with tf.name_scope('print'):
                    # Use a print node in the graph. The first input is the input data to pass to this node,
                    # the second is an array of which nodes in the graph you would like to print
                    # pr_mult = tf.Print(mult_y, [mult_y, newGrid], summarize = 25)
                    self.print_out = tf.print("\n Printing Tensors \n"
                                              "\n self.inhibCols = ", self.inhibCols,
                                              "\n self.notInhibOrActNum = ", self.notInhibOrActNum,
                                              "\n self.activeColumnVect = ", self.activeColumnVect,
                                              "\n self.activeColumnVectFinal = ", self.activeColumnVectFinal,
                                              summarize=200)

                # Make sure the print_out is performed during the graph execution.
                with tf.control_dependencies([self.print_out]):
                    # Perform some tf operation so the print out occurs.
                    self.inhibColsFinal_print = tf.multiply(self.inhibColsFinal, 1)


                # # Print the output.
                # with tf.name_scope('print'):
                #     # Use a print node in the graph. The first input is the input data to pass to this node,
                #     # the second is an array of which nodes in the graph you would like to print
                #     # pr_mult = tf.Print(mult_y, [mult_y, newGrid], summarize = 25)
                #     self.print_out = tf.print(self.inhibCols,
                #                               [self.inhibCols,
                #                                self.notInhibOrActNum,
                #                                self.colOverlapVect,
                #                                self.activeColumnVect],
                #                               message="Print", summarize=200)



            #self.summary_writer = tf.summary.FileWriter(self.logsPath, graph=tf.get_default_graph())

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

        # print "inputGrid = \n%s" % inputGrid
        width = len(inputGrid[0])
        height = len(inputGrid)

        rolledInputGrid = np.array([[0 for i in range(width)] for j in range(height)])
        outputGrid = np.array([[0 for i in range(width)] for j in range(height)])
        firstNonZeroIndVect = np.array([0 for i in range(width)])
        firstNonZeroVect = np.array([0 for i in range(width)])

        #print "width = %s height = %s" % (width, height)
        print("Setting Up tf inhibition calculator")
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
        print("Done")

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
                                     tf.cast(tf.greater(rolledInputGrid, 0), tf.int32),
                                     tf.cast(rolledInputGrid +
                                             tf.gather(firstNonZeroIndVect, self.col_num) -
                                             tf.gather(firstNonZeroVect, self.col_num) + 1,
                                             tf.int32)
                                    )

        # Run the tf graph
        with tf.Session() as sess:
            outputGrid = sess.run(convolePat)

        print("colInConvoleList = \n%s" % outputGrid)

        return outputGrid

    def calculateNonPaddingSumVect(self, inputGrid):

        with tf.name_scope('nonPaddingSumVect'):
            nonPaddingSumVect = tf.reduce_sum(
                                              tf.cast(tf.greater(inputGrid, 0), tf.float32),
                                              1)

        outputGrid = None
        # Run the tf graph
        with tf.Session() as sess:
            outputGrid = sess.run(nonPaddingSumVect)

        print("nonPaddingSumVect = \n%s" % outputGrid)

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
            colInhibInputs = self.poolInputs(inputGrid)

        # Run the tf graph
        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())

            inputInhibCols = sess.run(colInhibInputs)

        print("getColInhibInputs Input = \n%s" % inputGrid)
        print("inputInhibCols = \n%s" % inputInhibCols)
        #print("inputInhibCols.shape = %s,%s" % inputInhibCols.shape)
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
            kernel = [1, self.potentialHeight, self.potentialWidth, 1]
            # The image must be a 4d tensor for the convole function. Add some extra dimensions.
            image = tf.reshape(inputGrid, [1, self.height, self.width, 1], name='image')
            # Create the stride for the convolution
            stride = [1, self.stepY, self.stepX, 1]

            # Create the padding sizes to use and the padding node.
            with tf.name_scope('padding'):
                [padU, padD, padL, padR] = self.getPaddingSizes()
                # print("[padL, padR, padU, padD] = %s, %s, %s, %s" % (padL, padR, padU, padD))
                paddings = tf.constant([[0, 0], [padU, padD], [padL, padR], [0, 0]])
                # set the padding
                if self.wrapInput is True:
                    # We will add our own padding so we can reflect the input.
                    # This prevents the edges from being unfairly hindered due to a smaller overlap with the kernel.
                    paddedInput = tf.pad(image, paddings, "REFLECT")
                else:
                    paddedInput = tf.pad(image, paddings, "CONSTANT")

            # From the padded input for each HTM column get a list of the potential inputs that column can connect to.
            # Rates can be used to set how many pixels between each slected pixel are skipped.
            imageNeib4d = tf.extract_image_patches(images=paddedInput,
                                                   ksizes=kernel,
                                                   strides=stride,
                                                   rates=[1, 1, 1, 1],
                                                   padding='VALID')

            # print("kernel = \n%s" % kernel)
            # print("stride = \n%s" % stride)
            # print("image = \n%s" % image)
            # print("paddedInput = \n%s" % paddedInput)
            # print("imageNeib4d = \n%s" % imageNeib4d)

            # Reshape rates as the output is 3 dimensional and we only require 2 dimensions
            # A list for each of the htm columns that shows the results of each columns convolution
            imageNeib = tf.reshape(imageNeib4d,
                                   [self.numColumns,
                                    self.potentialHeight*self.potentialWidth],
                                   name='overlapImage')

            # tf.squeeze Removes dimensions of size 1 from the shape of a tensor.
            colConvole = tf.squeeze(imageNeib, name="colConvole")

            return colConvole

    def checkInhibCols(self, colOverlapMatOrig, colConvolePatternIndex, inhibCols):
        # Calculate if any column in the matrix is inhibited.
        # Any inhibited columns should be set as zero.
        # Any columns not inhibited should be set to the input matrix value.
        with tf.name_scope('check_inhibCols'):
            colConvolePatInd = tf.constant(self.colConvolePatternIndex, dtype=tf.int32, name='colConvolePatternIndex')
            col_con_pat_neg_one = tf.add(colConvolePatInd, -1, name='col_con_pat_neg_one')
            zeros_mat_int32 = tf.zeros_like(col_con_pat_neg_one, dtype=tf.int32, name='zeros_mat_int32')
            zeros_mat_f32 = tf.zeros_like(col_con_pat_neg_one, dtype=tf.float32, name='zeros_mat_f32')

            indicies = tf.maximum(col_con_pat_neg_one, zeros_mat_int32)
            cur_inhib_cols_sel = tf.gather(inhibCols, indicies, name='cur_inhib_cols_sel')

            set_winToZero = tf.where(tf.equal(cur_inhib_cols_sel, 1), zeros_mat_f32, colOverlapMatOrig)

            check_lZeroCol = tf.where(tf.greater_equal(col_con_pat_neg_one, zeros_mat_int32), set_winToZero, zeros_mat_f32)

            return check_lZeroCol

    def calculateActiveCol(self, colOverlapMat):
        # Calculate the active columns. The columns that have a higher overlap
        # value then the neighbouring ones.

        with tf.name_scope('get_activeCol'):
            numCols, numInhib = colOverlapMat.get_shape().as_list()
            #print("numCols = ", numCols)
            #print("numInhib = ", numInhib)

            # Create just a vector storing the row numbers for each column.
            # This is just an incrementing vector from zero to the number of columns - 1
            rowNumVect = tf.constant(self.row_numVect, dtype=tf.int32)

            # Sort the colOverlap matrix for each row. A row holds the inhib overlap
            # values for a single column.
            sortedColOverlapMat = tf.contrib.framework.sort(colOverlapMat,
                                                            axis=1,
                                                            direction='ASCENDING',
                                                            name='sortedColOverlapMat')

            # print("sortedColOverlapMat = \n%s" % sortedColOverlapMat)

            # Get the minLocalActivity for each col.
            colIndicies = tf.subtract(numInhib, self.minOverlapIndex)

            indicies = tf.stack([rowNumVect, colIndicies], axis=-1)
            minLocalAct = tf.gather_nd(sortedColOverlapMat, indicies)

            # Now calculate for each columns list of overlap values, which ones are larger
            # then the minLocalActivity number.
            # Broadcast the minLocalAct so it has the same dim as colOverlapMat
            # This is achieved by adding an extra empty dimension to the minLocalAct then broadcasting
            expMinLocalAct = tf.expand_dims(minLocalAct, axis=1, name='expMinLocalAct')
            bdim = [numCols, numInhib]
            #print("expMinLocalAct \n", expMinLocalAct)
            #print("bdim \n", bdim)
            minLocalActExpand = tf.broadcast_to(expMinLocalAct, bdim, name='minLocalActExpand')
            activeColsTemp = tf.cast(tf.greater(colOverlapMat, minLocalActExpand),
                                     tf.int32)
            activeCols = tf.cast(tf.greater(activeColsTemp, 0),
                                 tf.int32)

            #print("activeCols = \n%s" % activeCols)

            return activeCols

    def calculateActiveColumnVect(self, activeCols, inhibCols, colOverlapVect):
        # Calculate for each column a list of columns which that column can
        # be inhibited by. Set the winning columns in this list as one.
        # If a column is inhibited already then all those positions in
        # the colwinners relating to that col are set as one. This means
        # the inhibited columns don't determine the active columns

        # Calculate
        # a matrix of the columns which should stay active because they
        # won the inhibition convolution for all columns.
        # if a column is inhibited then set that location to one only if
        # that row does not represent that inhibited column.
        with tf.name_scope('get_activeColMat'):
            act_cols = tf.convert_to_tensor(activeCols, dtype=tf.int32, name='act_cols')
            col_pat = tf.convert_to_tensor(self.colInConvoleList, dtype=tf.int32, name='col_pat')
            col_num2 = tf.convert_to_tensor(self.col_num, dtype=tf.int32, name='col_num2')
            row_numMat4 = tf.convert_to_tensor(self.row_numMat, dtype=tf.int32, name='row_numMat4')
            cur_inhib_cols4 = tf.convert_to_tensor(inhibCols, dtype=tf.int32, name='cur_inhib_cols4')

            cur_inhib_cols_row = tf.gather(cur_inhib_cols4, row_numMat4, name='cur_inhib_cols_row')
            col_pat_neg_one = tf.add(col_pat, -1, name='col_pat_neg_one')

            zeros_mat = tf.zeros_like(cur_inhib_cols_row, dtype=tf.int32, name='zeros_mat')
            ones_mat = tf.ones_like(cur_inhib_cols_row, dtype=tf.int32, name='ones_mat')

            test_meInhib = tf.where(tf.equal(cur_inhib_cols_row, 1), zeros_mat, ones_mat, name='test_meInhib')
            indicies = tf.stack([tf.maximum(col_pat_neg_one, zeros_mat), col_num2], axis=-1)
            set_winners = tf.gather_nd(act_cols, indicies, name='set_winners')

            # Get the values at the non negative indicies. We get the value at index zero for the negative indicies.
            # These are not used.
            cur_inhib_col_pat = tf.gather(cur_inhib_cols4, tf.maximum(col_pat_neg_one, zeros_mat),
                                          name='cur_inhib_col_pat')

            check_colNotInhib = tf.where(tf.less(cur_inhib_col_pat, ones_mat), set_winners, test_meInhib,
                                         name='check_colNotInhib')
            check_colNotPad = tf.where(tf.greater(col_pat_neg_one, zeros_mat), check_colNotInhib, zeros_mat)

            colwinners = check_colNotPad

        # Now calculate which columns won enough of their colwinners list.
        # This creates a vector telling us which columns have the highest
        # overlap values and should be active.
        # The rows that have more then or equal to
        # the input nonPaddingSumVect. If this is true then set
        # in the output vector the col this row represents as active.
        # This function calculates if a column beat all the other non inhibited
        # columns in the convole overlap groups.
        with tf.name_scope('get_activeColVect'):
            # Make sure the self.nonPaddingSumVect is greater than zero.
            zeros_vec = tf.zeros_like(self.nonPaddingSumVect, dtype=tf.int32, name='zeros_vec')
            w_cols = tf.reduce_sum(colwinners, axis=1)
            test_lcol = tf.cast(tf.greater_equal(w_cols, self.nonPaddingSumVect), tf.int32, name='test_meInhib')
            test_meInhib = tf.where(tf.greater(self.nonPaddingSumVect, 0), test_lcol, zeros_vec, name='test_meInhib')
            # If the column has a zero overlap value (ie its overlap value
            # plus the tiebreaker is less then one then do not allow it to be active.
            activeColumnVect = tf.where(tf.less(colOverlapVect, 1), zeros_vec, test_meInhib)

        return activeColumnVect

    def calculateInhibCols(self, activeColumnVect, colOverlapVect):
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
            actColsVect = activeColumnVect
            colConvolePatInd = tf.constant(self.colConvolePatternIndex, dtype=tf.int32, name='colConvolePatternIndex')

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

            get_colsConPat = tf.where(tf.greater_equal(get_colsInVect, self.desiredLocalActivity), ones_mat, zeros_mat)
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
            colInConvoleList_negone = tf.add(self.colInConvoleList, -1)
            colin_numActColsInConVect = tf.gather(numActColsInConVect, tf.maximum(colInConvoleList_negone, zeros_mat))

            check_numActCols = tf.where(tf.greater_equal(colin_numActColsInConVect, self.desiredLocalActivity),
                                        ones_mat, zeros_mat)
            colin_actColVect = tf.gather(actColsVect, tf.maximum(colInConvoleList_negone, zeros_mat))
            check_colIndAct = tf.where(tf.greater(colin_actColVect, 0), check_numActCols, zeros_mat)

            get_rowActiveColumnVect = tf.gather(actColsVect, self.row_numMat)
            check_colsRowInAct = tf.where(tf.greater(get_rowActiveColumnVect, 0), zeros_mat, check_colIndAct)
            check_gtZero = tf.where(tf.greater(self.colInConvoleList, 0), check_colsRowInAct, zeros_mat)

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
            # Calculate if an input vector is larger then a scalar (element wise).
            inhibitedColsVect3 = tf.where(tf.greater(numActColsInConVect3, self.desiredLocalActivity), ones_vec, zeros_vec)
            # All three inhibitedColsVect vectors indicate that the column should be inhibited.
            # Add them together to find the total inhibited columns.
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
            tf_width = tf.constant(self.width, dtype=tf.int32, name='width')
            tf_height = tf.constant(self.height, dtype=tf.int32, name='height')
            red_inhibCols = tf.reduce_sum(inhibCols, 0)
            red_actColVect = tf.reduce_sum(actColsVect, 0)
            # Calculate the number of columsn that are not active and are not inhibited.
            # This number will be zero when the
            notInhibOrActNum = tf_width * tf_height - red_inhibCols - red_actColVect

        return inhibCols, notInhibOrActNum

    def body(self, inhibCols, notInhibOrActNum, activeColumnVect):
        # A function that is run in a tensorflow while loop

        # print("inhibCols = \n%s" % inhibCols)

        colOverlapMat = self.checkInhibCols(self.colOverlapMatOrig,
                                            self.colConvolePatternIndex,
                                            inhibCols)
        activeCols_new = self.calculateActiveCol(colOverlapMat)
        activeColumnVect_new = self.calculateActiveColumnVect(activeCols_new,
                                                              inhibCols,
                                                              self.colOverlapVect)
        inhibCols, notInhibOrActNum = self.calculateInhibCols(activeColumnVect_new,
                                                              self.colOverlapVect)

        return inhibCols, notInhibOrActNum, activeColumnVect_new

    def condition(self, inhibCols, notInhibOrActNum, activeColumnVect):
        # The condition to end the while loop.
        # This must return false for the while loop to end.
        # The calculation is complete when all columns are either inhbited or are active.

        # SHOULD BE THIS return notInhibOrActNum > 0
        #return notInhibOrActNum > 0
        return notInhibOrActNum < 1

    def calculateWinningCols(self, overlapsGrid=None):
        # Take the overlapsGrid and calculate a binary list
        # describing the active columns ( 1 is active, 0 not active).

        # Run the Graph
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Merge all the summaries into one tensorflow operation
            merge = tf.summary.merge_all()

            if overlapsGrid is not None:
                inhibitedCols, notInhibOrActNum = sess.run([self.inhibColsFinal_print, self.notInhibOrActNumFinal],
                                                           feed_dict={self.colOverlapsGrid: overlapsGrid})
            else:
                inhibitedCols, notInhibOrActNum = sess.run([self.inhibColsFinal, self.notInhibOrActNumFinal])

            self.summary_writer = tf.summary.FileWriter(self.logsPath, graph=tf.get_default_graph())

            # print("tf_activeColumns = \n%s" % tf_activeColumns)
            # print("inhibitedCols = \n%s" % inhibitedCols)
            print("inhibitedCols, notInhibOrActNum = %s, %s" % (inhibitedCols, notInhibOrActNum))

            print("\nRun the command line:\n"
                  "--> tensorboard --logdir=/tmp/tensorflow_logs "
                  "\nThen open http://0.0.0.0:6006/ into your web browser")

        # activeColumnVect = tf_activeColumns

        return inhibitedCols

if __name__ == '__main__':

    potWidth = 3
    potHeight = 3
    centerInhib = 1
    numRows = 4
    numCols = 4
    desiredLocalActivity = 2
    minOverlap = 1

    # Some made up inputs to test with
    #colOverlapsGrid = np.random.randint(1, size=(numRows, numCols))
    # colOverlapsGrid = np.array([[8, 4, 5, 8],
    #                            [8, 6, 1, 6],
    #                            [7, 7, 9, 4],
    #                            [2, 3, 1, 5]])
    colOverlapsGrid = np.array([[9, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 9, 0],
                               [0, 0, 0, 0]])
    print("colOverlapsGrid = \n%s" % colOverlapsGrid)

    inhibCalculator = inhibitionCalculator(numCols, numRows,
                                           potWidth, potHeight,
                                           desiredLocalActivity,
                                           minOverlap,
                                           centerInhib)

    #cProfile.runctx('activeColumns = inhibCalculator.calculateWinningCols(colOverlapsGrid)', globals(), locals())
    activeColumns = inhibCalculator.calculateWinningCols(colOverlapsGrid)

    #activeColumns = activeColumns.reshape((numRows, numCols))
    #print "colOverlapsGrid = \n%s" % colOverlapGrid
    #print "activeColumns = \n%s" % activeColumns







