import tensorflow as tf
import numpy as np
import math
import random

from datetime import datetime


'''
A class used to calculate the overlap values for columns
in a single HTM layer. This class uses tensorflow functions
to speed up the computation. Can be run on a CPU or GPU.

Take a numpy input 2D matrix and convert this into a
tensor that holds inputs that are connected
by potential synapses to columns.

Eg
input = [[0,0,1,0]
         [1,0,0,0]
         [0,1,0,1]
         [0,0,0,0]
         [0,1,0,0]]

Output = [[x1,x2,x3,x4]
          [x5,x6,x7,x8]
          [x9,x10,x11,x12]
          [x13,x14,x15,x16]
          [x17,x18,x19,x20]]

x10 = [1,0,0,0,1,0,0,0,0]

potential_width = 3
potential_height = 3

Take this output and calcuate the overlap for each column.
This is the sum of 1's for each columns input.

'''


class OverlapCalculator():
    def __init__(self, potentialWidth, potentialHeight,
                 columnsWidth, columnsHeight,
                 inputWidth, inputHeight,
                 centerPotSynapses, connectedPerm,
                 minOverlap, wrapInput, numInputs=1):

        # Overlap Parameters
        ###########################################
        # Specifies if the potential synapses are centered
        # over the columns
        self.centerPotSynapses = centerPotSynapses
        # Use a wrap input function instead of padding the input
        # to calculate the overlap scores.
        self.wrapInput = wrapInput
        self.potentialWidth = potentialWidth
        self.potentialHeight = potentialHeight
        self.numPotSyn = self.potentialWidth * self.potentialHeight
        self.connectedPermParam = connectedPerm
        self.minOverlap = minOverlap
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        # How many inputs come in each timestep. If this is 3 then the new inputs
        # should contain 3 new input grids.
        self.numInputs = numInputs
        # Calculate how many columns are expected from these
        # parameters.
        self.columnsWidth = columnsWidth
        self.columnsHeight = columnsHeight
        self.numColumns = columnsWidth * columnsHeight
        # Store the potential inputs to every column.
        # Each row represents the inputs a columns potential synapses cover.
        self.colInputPotSyn = None
        # Store the potential overlap values for every column
        self.colPotOverlaps = None
        # StepX and Step Y describe how far each
        # columns potential synapses differ from the adjacent
        # columns in the X and Y directions. These parameters can't
        # change as tensorflow uses them to setup functions.
        self.stepX, self.stepY = self.getStepSizes(inputWidth, inputHeight,
                                                   self.columnsWidth, self.columnsHeight,
                                                   self.potentialWidth, self.potentialHeight)
        # Contruct a tiebreaker matrix for the columns potential synapses.
        # It contains small values that help resolve any ties in potential
        # overlap scores for columns.
        self.potSynTieBreaker = np.array([[0.0 for i in range(self.potentialHeight*self.potentialWidth)]
                                         for j in range(self.numColumns)])
        #import ipdb; ipdb.set_trace()
        self.makePotSynTieBreaker(self.potSynTieBreaker)
        # Store the potential inputs to every column plus the tie breaker value.
        # Each row represents the inputs a columns potential synapses cover.
        self.colInputPotSynTie = np.array([[0.0 for i in range(self.potentialHeight*self.potentialWidth)]
                                          for j in range(self.numColumns)])
        self.colTieBreaker = np.array([0.0 for i in range(self.numColumns)])
        self.makeColTieBreaker(self.colTieBreaker)

        # Create tensorflow variables and functions
        ############################################

        # Resets tensorflow graph
        tf.reset_default_graph()

        # Set the location to store the tensflow logs so we can run tensorboard.
        self.logsPath = '/tmp/tensorflow_logs/example/tf_htm/'
        now = datetime.now()
        self.logsPath = self.logsPath + now.strftime("%Y%m%d-%H%M%S") + "/"

        with tf.name_scope('inputs'):
            # A tensor representing the permanences of column synapses
            self.colSynPerm = tf.placeholder(tf.float32, [self.numColumns, self.numPotSyn], name='colSynPerm')
            # A variable to hold a tensor storing possibly multiple new input tensors (grids).
            self.inGrids = tf.placeholder(tf.float32, [self.numInputs, self.inputHeight, self.inputWidth], name='InputGrids')
            # variable to store just one new input tensor (one grid).
            self.newGrid = tf.placeholder(tf.float32, [1, self.inputHeight, self.inputWidth], name='NewGrid')
            # Using a placeholder create a dataset to store the new inputs in
            dataset = tf.data.Dataset.from_tensor_slices(self.inGrids)
            # Create an iterator to iterate through our dataset.
            self.iter = dataset.make_initializable_iterator()
            # Make the newGrid get just one grid from the dataset
            self.newGrid = self.iter.get_next()
            # A tensor full of zeros
            self.zerosMat = tf.zeros([self.potentialHeight*self.potentialWidth, self.numColumns], name='zerosMat')

        # Create the potential pool of inputs each column in the HTM could conenct to.
        with tf.name_scope('getColInputs'):

            # Create the variables to use in the convolution.
            with tf.name_scope('inputNeighbours'):
                # A variable to store the kernal used to workout the potential pool for each column
                # The kernal must be a 4d tensor.
                self.kernel = [1, self.potentialHeight, self.potentialWidth, 1]
                # The image must be a 4d tensor for the convole function. Add some extra dimensions.
                self.image = tf.reshape(self.newGrid, [1, self.inputHeight, self.inputWidth, 1], name='image')
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
                # A list for each of the htm columns that shows what each columns can potentially connect to.
                self.imageNeib = tf.reshape(self.imageNeib4d,
                                            [self.numColumns,
                                             self.potentialHeight*self.potentialWidth],
                                            name='overlapImage')

                # tf.squeeze Removes dimensions of size 1 from the shape of a tensor.
                self.colInputPotSyn = tf.squeeze(self.imageNeib)

        # Add a masked small tiebreaker value to the self.colInputPotSyn scores.
        with tf.name_scope('maskTiebreaker'):
            # Multiply the tiebreaker values by the colInputPotSyn then add them to it.
            # Since the colInputPotSyn contains ones and zeros some tiebreaker values are
            # masked out. This means the tie breaker will be different for each input
            # pattern.
            # print("colInputPotSyn.shape = %s" % self.colInputPotSyn.shape)
            # print("potSynTieBreaker.shape = %s,%s" % self.potSynTieBreaker.shape)
            self.potSynTieBreakerTf = tf.convert_to_tensor(self.potSynTieBreaker, np.float32, name='potSynTieBreaker')
            self.maskedTieBreaker = tf.multiply(self.colInputPotSyn, self.potSynTieBreakerTf)
            self.colInputPotSynTie = tf.add(self.colInputPotSyn, self.maskedTieBreaker)

        # Calculate the potential overlap scores for every column.
        # Sum the potential inputs for every column.
        with tf.name_scope('potOverlap'):
            self.colPotOverlaps = tf.reduce_sum(self.colInputPotSynTie, 1)

        # Call the tf functions to calculate the overlap value.
        with tf.name_scope('overlap'):
            # Each synapse is check to see if it is connected.
            # If so then the input that synapse connects to is returned, zero otherwise.
            # We have to cast the output of the comparison from a bool to a float so it can be multiplied.
            self.connectedSynInputs = tf.multiply(
                                                  tf.cast(
                                                          tf.greater(self.colSynPerm, self.connectedPermParam),
                                                          tf.float32),
                                                  self.colInputPotSyn)
            # Compute the sum of all the inputs that are connected to a synapses. This is the overlap score of an HTM column.
            # This is done over the first dimension to sum all the synapses for one column.
            self.colOverlapVals = tf.reduce_sum(self.connectedSynInputs, 1)
            # Add a small tiebreaker value to the column overlap scores vector.
            self.colOverlapValsTie = tf.add(self.colOverlapVals, self.colTieBreaker)

        # Set any overlap values that are smaller then the
        # minOverlap value to zero.
        with tf.name_scope('removeSmallOverlaps'):
            self.colOverlapValsTieMin = tf.multiply(
                                                    tf.cast(tf.greater(self.colOverlapValsTie,
                                                            tf.cast(self.minOverlap, tf.float32)),
                                                            tf.float32),
                                                    self.colOverlapValsTie)

        # Create variables to store certain tensors in our graph.
        with tf.variable_scope("storedVars1", reuse=tf.AUTO_REUSE):
            # We set these variables as non trainable since we are not using back propagation.
            self.prevColInputPotSyn = tf.get_variable("prevColInputPotSyn",
                                                      shape=self.colInputPotSyn.get_shape().as_list(),
                                                      dtype=tf.float32,
                                                      initializer=tf.zeros_initializer,
                                                      trainable=False)
            self.prevColInputPotSyn = self.prevColInputPotSyn.assign(self.colInputPotSyn)

            # We set these variables as non trainable since we are not using back propagation.
            self.prevColSynPerm = tf.get_variable("prevColSynPerm",
                                                  shape=self.colSynPerm.get_shape().as_list(),
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer,
                                                  trainable=False)
            self.prevColSynPerm = self.prevColSynPerm.assign(self.colSynPerm)

            self.prevColPotOverlaps = tf.get_variable("prevColPotOverlaps",
                                                      shape=self.colPotOverlaps.get_shape().as_list(),
                                                      dtype=tf.float32,
                                                      initializer=tf.zeros_initializer,
                                                      trainable=False)
            self.prevColPotOverlaps = self.prevColPotOverlaps.assign(self.colPotOverlaps)

            self.prevOverlap = tf.get_variable("prevOverlap",
                                               shape=self.colOverlapVals.get_shape().as_list(),
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer,
                                               trainable=False)
            self.prevOverlap = self.prevOverlap.assign(self.colOverlapVals)

            self.prevOverlapTieMin = tf.get_variable("prevOverlapTieMin",
                                                     shape=self.colOverlapValsTieMin.get_shape().as_list(),
                                                     dtype=tf.float32,
                                                     initializer=tf.zeros_initializer,
                                                     trainable=False)
            self.prevOverlapTieMin = self.prevOverlapTieMin.assign(self.colOverlapValsTieMin)

        # Print the output.
        with tf.name_scope('print'):
            # Use a print node in the graph. The first input is the input data to pass to this node,
            # the second is an array of which nodes in the graph you would like to print
            # pr_mult = tf.Print(mult_y, [mult_y, newGrid], summarize = 25)
            self.pr_mult = tf.Print(self.colOverlapValsTieMin,
                                    [self.colInputPotSyn, self.colPotOverlaps, self.colOverlapValsTieMin],
                                    message="Print", summarize=200)

        # Create a summary to monitor padded input tensor
        tf.summary.histogram("paddedInput", self.paddedInput)
        # Create images from the  inputs to display in tensorboard
        with tf.name_scope('tbImages'):
            self.InputFloat = tf.to_float(self.image)
            tf.summary.image("InputImage", self.InputFloat)
            self.paddedInputFloat = tf.to_float(self.paddedInput)
            tf.summary.image("paddedInputImage", self.paddedInputFloat)
            #self.colInputPotSynFloat = tf.to_float(self.imageNeib4d)
            #tf.summary.image("colInputPotSyn", self.colInputPotSynFloat)

        # op to write logs to Tensorboard
        self.summary_writer = tf.summary.FileWriter(self.logsPath, graph=tf.get_default_graph())

        ###################################################################
        # Create the graph to calculate the indicies that the potential Synapses connect to.
        ###################################################################
        # First create a matrix with the same dimensions as
        # the inputGrid but where each position holds an index.
        # The index is just a number representing that element in the inputGrid.
        # Start the index at one not zero. This is so later we can minus one so all
        # the padding elements will have a value of -1 and are easily identifiable.
        self.tempIndexInputGrid = np.array([[[1+i + j*inputWidth for i in range(inputWidth)]
                                            for j in range(inputHeight)]
                                            for k in range(numInputs)], dtype=float)


        # # Take the input and put it into a 4D tensor.
        # # This is because the tensorflow convolution works with 4D tensors only.
        # newShape = np.insert(self.tempIndexInputGrid.shape, 0, 1)
        # newShape = np.insert(newShape, len(newShape), 1)
        # self.tempIndexInputGrid = np.reshape(self.tempIndexInputGrid, newShape)
        print("self.tempIndexInputGrid.shape = \n%s self.tempIndexInputGrid = \n%s" % (self.tempIndexInputGrid.shape, self.tempIndexInputGrid))
        # self.indexInputGrid = None
        # self.indexInputGridAdd = None
        # self.indexInputGridZeroPad = None
        # self.indexInputGridPad = None
        # self.imagePotind = None
        # self.inputPotSynIndex = None
        # self.prevInputPotSynIndex = None
        # self.indexInputGridPadImage = None
        #self.createGetPotSynPosGraph()

    def makePotSynTieBreaker(self, tieBreaker):
        # create a tie breaker matrix holding small values for each element in
        # the self.colInputPotSyn grid. The tie breaker values are created such that
        # for a particular row in the colInputPotSyn adding all tie breaker values up
        # the result is less then 1. We will make it less then 0.5. THe tie breaker values
        # are all multiples of the same number. Each row in the colInputPotSyn grid
        # has a different pattern of tie breaker values. THis is done by sliding the previous
        # rows values along by 1 and wrapping at the end of the row.
        # They are used  to resolve situations where columns have the same overlap number.

        inputHeight = len(tieBreaker)
        inputWidth = len(tieBreaker[0])

        # Use the sum of all integer values less then or equal to formula.
        # This is because each row has its tie breaker values added together.
        # We want to make sure the result from adding the tie breaker values is
        # less then 0.5 but more then 0.0.
        n = float(inputWidth)
        normValue = float(0.5/(n*(n+1.0)/2.0))
        #normValue = 1.0/float(2*inputWidth+2)
        print("maxNormValue = %s" % (n*(n+1.0)/2.0))
        print("normValue = %s" % normValue)
        print("tie Width = %s" % inputWidth)

        rowsTie = np.arange(inputWidth)+1
        rowsTie = rowsTie*normValue

        # Use a seeded random sample of the above array
        seed = 1
        random.seed(seed)

        #import ipdb; ipdb.set_trace()
        # Create a tiebreaker that changes for each row.
        for j in range(len(tieBreaker)):
            tieBreaker[j] = random.sample(rowsTie, len(rowsTie))
        #print("tieBreaker = \n%s" % tieBreaker)

    def makeColTieBreaker(self, tieBreaker):
        # Make a vector of tiebreaker values to add to the columns overlap values vector.
        normValue = 1.0/float(2*self.numColumns+2)

        # Initialise a random seed so we can get the same random numbers.
        # This means the tie breaker will be the same each time but it will
        # be randomly distributed over the cells.
        seed = 1
        random.seed(seed)

        # Create a tiebreaker that is not biased to either side of the columns grid.
        for j in range(len(tieBreaker)):
            # The tieBreaker is a flattened vector of the columns overlaps.

            # workout the row and col number of the non flattened matrix.
            rowNum = math.floor(j/self.columnsWidth)
            colNum = j % self.columnsWidth
            if (random.random() > 0.5) == 1:
                # Some positions are bias to the bottom left
                tieBreaker[j] = ((rowNum+1)*self.columnsWidth+(self.columnsWidth-colNum-1))*normValue
            else:
                # Some Positions are bias to the top right
                tieBreaker[j] = ((self.columnsHeight-rowNum)*self.columnsWidth+colNum)*normValue

    def getStepSizes(self, inputWidth, inputHeight, colWidth, colHeight, potWidth, potHeight):
        # Work out how large to make the step sizes so all of the
        # inputGrid can be covered as best as possible by the columns
        # potential synapses.

        stepX = int(round(float(inputWidth)/float(colWidth)))
        stepY = int(round(float(inputHeight)/float(colHeight)))

        #import ipdb; ipdb.set_trace()
        # The step sizes may need to be increased if the potential sizes are too small.
        if potWidth + (colWidth-1)*stepX < inputWidth:
            # Calculate how many of the input elements cannot be covered with the current stepX value.
            uncoveredX = (inputWidth - (potWidth + (colWidth - 1) * stepX))
            # Use this to update the stepX value so all input elements are covered.
            stepX = stepX + int(math.ceil(float(uncoveredX) / float(colWidth-1)))

        if potHeight + (colHeight-1)*stepY < self.inputHeight:
            uncoveredY = (inputHeight - (potHeight + (colHeight - 1) * stepY))
            stepY = stepY + int(math.ceil(float(uncoveredY) / float(colHeight-1)))

        return stepX, stepY

    def checkNewInputParams(self, newColSynPerm, newInput):
        # Check that the new input has the same dimensions as the
        # originally defined input parameters.
        # print("self.numInputs = %s len(newInput) = %s" %(self.numInputs, len(newInput)))
        # print("newInput = %s" %newInput)
        assert self.numInputs == len(newInput)
        # print("self.inputHeight = %s len(newInput[0]) = %s" %(self.inputHeight, len(newInput[0])))
        assert self.inputHeight == len(newInput[0])
        assert self.inputWidth == len(newInput[0][0])
        assert self.potentialWidth * self.potentialHeight == len(newColSynPerm[0])
        # Check the number of rows in the newColSynPerm matrix equals
        # the number of expected columns.
        assert self.numColumns == len(newColSynPerm)

    def getPaddingSizes(self):
        # Calculate the amount of padding to add to each dimension.
        topPos_y = 0
        bottomPos_y = 0
        leftPos_x = 0
        rightPos_x = 0

        # This calculates how much of the input grid is not covered by
        # the htm grid in each dimension using the step sizes.
        leftOverWidth = self.inputWidth - (1 + (self.columnsWidth - 1) * self.stepX)
        leftOverHeight = self.inputHeight - (1 + (self.columnsHeight - 1) * self.stepY)

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

    def addPaddingToInput(self, inputGrid, useZeroPadVal=True):
        # Add padding elements to the input Grid so that the
        # convole function can convole over the input.
        # Padding value.
        [topPos_y, bottomPos_y, leftPos_x, rightPos_x] = self.getPaddingSizes()
        if useZeroPadVal is False:
            padValue = -1
        else:
            padValue = 0

        # Add the padding around the edges of the inputGrid
        inputGrid = np.lib.pad(inputGrid,
                               ((0, 0),
                                (0, 0),
                                (topPos_y, bottomPos_y),
                                (leftPos_x, rightPos_x)),
                               'constant',
                               constant_values=(padValue))

        # print("inputGrid = \n%s" % inputGrid)

        return inputGrid

    def addTfPad(self, image, wrapInput):
        # Create the padding sizes to use and the padding node.
        paddedInput = None
        with tf.name_scope('pad'):
            [padU, padD, padL, padR] = self.getPaddingSizes()
            # print("[padL, padR, padU, padD] = %s, %s, %s, %s" % (padL, padR, padU, padD))
            padDim = tf.constant([[0, 0], [padU, padD], [padL, padR], [0, 0]])
            # set the padding
            if wrapInput is True:
                # This prevents the edges from being unfairly hindered due to a smaller overlap with the kernel.
                paddedInput = tf.pad(image, padDim, "REFLECT")
            else:
                paddedInput = tf.pad(image, padDim, "CONSTANT")
        return paddedInput

    def getPotShape(self):
        # Return the potential width and height.
        return self.potentialWidth, self.potentialHeight

    def getPotentialSynapsePos(self, inputWidth, inputHeight):
        # Creates and runs a new tf graph to return a matrix of index positions
        # that each column potential synpases connects to in the input.

        # Run the tf graph
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Merge all the summaries into one tensorflow operation
            merge = tf.summary.merge_all()

            # feed the placeholder with data. This puts the next input into our graph.
            # The feed_dict is a dictionary holding the placeholders with the real data
            sess.run(self.iter.initializer, feed_dict={self.inGrids: self.tempIndexInputGrid})

            for i in range(self.numInputs):

                summary, potInd = sess.run([merge, self.colInputPotSyn])
                                           #feed_dict={self.colSynPerm: colSynPerm})

            # Now turn the inputPotSynIndex into a matrix holding the X and Y indicies
            # for the element in the inpuGrid that a potential synapse connects to.
            # First minus one so all the padded elements get a value of minus one and are easily identifiable.
            potIndRes = potInd-1
            print("potIndRes = \n%s" % potIndRes)
            potSynXYIndex = (np.floor_divide(potIndRes, inputWidth),
                             np.remainder(potIndRes, inputWidth))
            print("potSynXYIndex X = \n")
            print(potSynXYIndex[0])
            print("potSynXYIndex Y = \n")
            print(potSynXYIndex[1])
            #Write logs at every iteration
            self.summary_writer.add_summary(summary)

        return potSynXYIndex

    def calculateOverlap(self, colSynPerm, inputGrid):

        # Check that the new inputs are the same dimensions as the old ones
        # and the colsynPerm match the original specified parameters.
        self.checkNewInputParams(colSynPerm, inputGrid)

        # Calculate the inputs to each column
        #import ipdb; ipdb.set_trace()
        # self.colInputPotSyn = self.getColInputs(inputGrid)

        # Calculate the inputs to each column

        # Run the Graph
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Merge all the summaries into one tensorflow operation
            merge = tf.summary.merge_all()

            # feed the placeholder with data. This puts the next input into our graph.
            # The feed_dict is a dictionary holding the placeholders with the real data
            sess.run(self.iter.initializer, feed_dict={self.inGrids: inputGrid})

            for i in range(self.numInputs):
                print("\nNew overlap calculation")

                # Run the tensor flow overlap graph and get the summary for tensorboard.
                summary, overlap, colInPotSyn = sess.run([merge,
                                                          self.pr_mult,
                                                          self.prevColInputPotSyn],
                                                         feed_dict={self.colSynPerm: colSynPerm})

                print("overlap plus tiebreaker = \n%s" % overlap.reshape((self.columnsWidth, self.columnsHeight)))

                # Write logs at every iteration
                self.summary_writer.add_summary(summary)

            print("\nRun the command line:\n"
                  "--> tensorboard --logdir=/tmp/tensorflow_logs "
                  "\nThen open http://0.0.0.0:6006/ into your web browser")


        return overlap, colInPotSyn


if __name__ == '__main__':

    potWidth = 3
    potHeight = 3
    centerPotSynapses = 1
    numInputRows = 3
    numInputCols = 3
    numColumnRows = 3
    numColumnCols = 3
    numInputs = 1
    connectedPerm = 0.3
    minOverlap = 2
    numPotSyn = potWidth * potHeight
    numColumns = numColumnRows * numColumnCols
    wrapInput = False

    # Create an array representing the permanences of colums synapses
    colSynPerm = np.random.rand(numColumns, numPotSyn)
    # colSynPerm = np.zeros((numColumns, numPotSyn))
    # colSynPerm = np.add(colSynPerm,1)
    # Print the synapses where the synapse is connected
    # print("colSynPerm = \n%s" % colSynPerm)
    conColSynPerm = np.array(np.where(colSynPerm > connectedPerm, 1, 0))
    print("conColSynPerm = \n%s" % conColSynPerm)


    # Create an instance of the overlap calculation class
    overlapCalc = OverlapCalculator(potWidth,
                                    potHeight,
                                    numColumnCols,
                                    numColumnRows,
                                    numInputCols,
                                    numInputRows,
                                    centerPotSynapses,
                                    connectedPerm,
                                    minOverlap,
                                    wrapInput,
                                    numInputs)

    newInputMat = np.array(np.random.randint(2, size=(numInputs, numInputRows, numInputCols)))
    print("newInputMat = \n%s" % newInputMat)


    # Return both the overlap values and the inputs from
    # the potential synapses to all columns.
    colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)
    #print( "len(colOverlaps) = %s" % len(colOverlaps))
    #print( "colOverlaps = \n%s" % colOverlaps)
    #print("colPotInputs = \n%s" % colPotInputs)

    # limit the overlap values so they are larger then minOverlap
    #colOverlaps = overlapCalc.removeSmallOverlaps(colOverlaps)

    #print("colOverlaps = \n%s" % colOverlaps)
