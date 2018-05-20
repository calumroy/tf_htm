import tensorflow as tf
import numpy as np
import math
import random


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
                 minOverlap, wrapInput):

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
        self.connectedPermParam = connectedPerm
        self.minOverlap = minOverlap
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        # Calculate how many columns are expected from these
        # parameters.
        self.columnsWidth = columnsWidth
        self.columnsHeight = columnsHeight
        self.numColumns = columnsWidth * columnsHeight
        # Store the potetnial inputs to every column.
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

        # Create tf variables and functions
        ############################################


        self.logs_path = '/tmp/tensorflow_logs/example/tf_htm/'

        np_array_1 = np.random.rand(3, 2)
        np_array_2 = np.random.rand(3, 2)

        self.y = tf.multiply(np_array_1, np_array_2)






    def makePotSynTieBreaker(self, tieBreaker):
        # create a tie breaker matrix holding small values for each element in
        # the self.colInputPotSyn grid. The tie breaker values are created such that
        # for a particular row in the colInputPotSyn adding all tie breaker values up
        # the result is less then 1. We will make it less then 0.5. THe tie breaker values
        # are all multiples of the same number. Each row in the colInputPotSyn grid
        # has a different pattern of tie breaker values. THis is done by sliding the previous
        # rows values along by 1 and wrapping at the end of the row.
        # They are used  to resolve situations where columns have the same overlap number.
        # The purpose of
        inputHeight = len(tieBreaker)
        inputWidth = len(tieBreaker[0])

        # Use the sum of all integer values less then or equal to formula.
        # This is because each row has its tie breaker values added together.
        # We want to make sure the result from adding the tie breaker values is
        # less then 0.5 but more then 0.0.
        n = float(inputWidth)
        normValue = float(0.5/(n*(n+1.0)/2.0))
        #normValue = 1.0/float(2*inputWidth+2)
        print "maxNormValue = %s" % (n*(n+1.0)/2.0)
        print "normValue = %s" % normValue
        print "tie Width = %s" % inputWidth

        rowsTie = np.arange(inputWidth)+1
        rowsTie = rowsTie*normValue

        # Use a seeded random sample of the above array
        seed = 1
        random.seed(seed)

        #import ipdb; ipdb.set_trace()
        # Create a tiebreaker that changes for each row.
        for j in range(len(tieBreaker)):
            tieBreaker[j] = random.sample(rowsTie, len(rowsTie))
        # print "self.tieBreaker = \n%s" % self.tieBreaker

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
        assert self.inputWidth == len(newInput[0])
        assert self.inputHeight == len(newInput)
        assert self.potentialWidth * self.potentialHeight == len(newColSynPerm[0])
        # Check the number of rows in the newColSynPerm matrix equals
        # the number of expected columns.
        assert self.numColumns == len(newColSynPerm)

    def getColInputs(self, inputGrid):
        # This function uses theano's convolution function to
        # return the inputs that each column potentially connects to.

        # It ouputs a matrix where each row represents the potential pool of inputs
        # that one column in a layer can connect too.

        # Take the input and put it into a 4D tensor.
        # This is because the theano function images2neibs
        # works with 4D tensors only.
        inputGrid = np.array([[inputGrid]])

        # print "inputGrid.shape = %s,%s,%s,%s" % inputGrid.shape
        firstDim, secondDim, width, height = inputGrid.shape

        # If we are not using the wrap input function then padding will need to be added
        # to the edges.
        if self.wrapInput == False:
            # work out how much padding is needed on the borders
            # using the defined potential width and potential height.
            inputGrid = self.addPaddingToInput(inputGrid)

        # print "padded InputGrid = \n%s" % inputGrid
        # print "inputGrid.shape = %s,%s,%s,%s" % inputGrid.shape
        # print "self.potentialWidth = %s" % self.potentialWidth
        # print "self.potentialHeight = %s" % self.potentialHeight
        # print "self.stepX = %s, self.stepY = %s" % (self.stepX, self.stepY)
        # Calculate the inputs to each column.
        if self.wrapInput == True:
            # Wrap the input grid to create the potential pool for each column.
            inputConPotSyn = self.pool_inputs_wrap(inputGrid)
            # Since we are wrapping and no padding was added to the input then
            # the inputConPotSyn could have more potential pool groups then the
            # number of columns in the layer. Delete the last rows of the inputConPotSyn.
            if len(inputConPotSyn) > self.numColumns:
                diff = len(inputConPotSyn) - self.numColumns
                num_pools = len(inputConPotSyn)
                inputConPotSyn = np.delete(inputConPotSyn, range(num_pools-diff,num_pools), axis=0)
        else:
            # Don't wrap the input. The columns on the edges have a smaller potential input pool.
            inputConPotSyn = self.pool_inputs(inputGrid)
        # The returned array is within a list so just use pos 0.
        # print "inputConPotSyn = \n%s" % inputConPotSyn
        # print "inputConPotSyn.shape = %s,%s" % inputConPotSyn.shape
        return inputConPotSyn


    def calculateOverlap(self, colSynPerm, inputGrid):

        # Check that the new inputs are the same dimensions as the old ones
        # and the colsynPerm match the original specified parameters.
        self.checkNewInputParams(colSynPerm, inputGrid)
        # Calcualte the inputs to each column
        #import ipdb; ipdb.set_trace()
        self.colInputPotSyn = self.getColInputs(inputGrid)

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(self.y)

            print("Y tensor values after calculation \n")
            print(self.y.eval())

            # op to write logs to Tensorboard
            summary_writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())

            # Write logs at every iteration
            #summary_writer.add_summary(summary, epoch * total_batch + i)

            print("\nRun the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")




        self.colInputPotSyn = colSynPerm
        colOverlapVals = inputGrid

        #print "colOverlapVals = \n%s" % colOverlapVals.reshape((self.columnsWidth, self.columnsHeight))
        return colOverlapVals, self.colInputPotSyn




if __name__ == '__main__':

    potWidth = 3
    potHeight = 3
    centerPotSynapses = 1
    numInputRows = 4
    numInputCols = 5
    numColumnRows = 4
    numColumnCols = 4
    connectedPerm = 0.3
    minOverlap = 3
    numPotSyn = potWidth * potHeight
    numColumns = numColumnRows * numColumnCols
    wrapInput = True

    # Create an array representing the permanences of colums synapses
    colSynPerm = np.random.rand(numColumns, numPotSyn)
    # To get the above array from a htm use
    # allCols = self.htm.regionArray[0].layerArray[0].columns.flatten()
    # colPotSynPerm = np.array([[allCols[j].potentialSynapses[i].permanence for i in range(36)] for j in range(1600)])

    print "colSynPerm = \n%s" % colSynPerm
    newInputMat = np.random.randint(2, size=(numInputRows, numInputCols))

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
                                    wrapInput)

    print "newInputMat = \n%s" % newInputMat

    # Return both the overlap values and the inputs from
    # the potential synapses to all columns.
    colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)
    print "len(colOverlaps) = %s" % len(colOverlaps)
    print "colOverlaps = \n%s" % colOverlaps
    print "colPotInputs = \n%s" % colPotInputs

    # limit the overlap values so they are larger then minOverlap
    colOverlaps = overlapCalc.removeSmallOverlaps(colOverlaps)

    print "colOverlaps = \n%s" % colOverlaps
