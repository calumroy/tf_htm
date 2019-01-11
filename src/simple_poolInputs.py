from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
from datetime import datetime


class poolInputs():
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
        # Set the location to store the tensflow logs so we can run tensorboard.
        self.logsPath = '/tmp/tensorflow_logs/example/tf_htm/'
        now = datetime.now()
        self.logsPath = self.logsPath + now.strftime("%Y%m%d-%H%M%S") + "/"

        # Take the colOverlapMat and add a small number to each overlap
        # value based on that row and col number. This helps when deciding
        # how to break ties in the inhibition stage. Note this is not a random value!
        # Make sure the tiebreaker contains values less then 1.
        normValue = 1.0/float(self.numColumns+1)
        self.tieBreaker = np.array([[(1+i+j*self.width)*normValue for i in range(self.width)] for j in range(self.height)])
        print("self.tieBreaker = \n%s" % self.tieBreaker)

        # A variable to hold a tensor storing the overlap values for each column
        self.colOverlapsGrid = tf.placeholder(tf.float32, [self.height, self.width], name='colOverlapsGrid')

        with tf.name_scope('addTieBreaker'):
            # Add the tieBreaker value to the overlap values.
            self.overlapsGridTie = tf.add(self.colOverlapsGrid, self.tieBreaker)

        with tf.name_scope('getColOverlapMatOrig'):
            self.colOverlapMatOrig = self.poolInputs(self.overlapsGridTie)

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

    def calculatePoolInputs(self, overlapsGrid=None):
        # Take the overlapsGrid and calculate a binary list
        # describing the active columns ( 1 is active, 0 not active).

        # Run the Graph
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Merge all the summaries into one tensorflow operation
            merge = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.logsPath, graph=tf.get_default_graph())

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            pooledOutputs = sess.run([self.colOverlapMatOrig],
                                     feed_dict={self.colOverlapsGrid: overlapsGrid},
                                     options=run_options,
                                     run_metadata=run_metadata)

            i = 1
            self.summary_writer.add_run_metadata(run_metadata, 'step%d' % i)

            return pooledOutputs


if __name__ == '__main__':

    potWidth = 30
    potHeight = 30
    centerInhib = 1
    numRows = 800
    numCols = 800
    desiredLocalActivity = 2
    minOverlap = 1

    # Some made up inputs to test with
    colOverlapsGrid = np.random.randint(10, size=(numRows, numCols))
    # colOverlapsGrid = np.array([[8, 4, 5, 8],
    #                            [8, 6, 1, 6],
    #                            [7, 7, 9, 4],
    #                            [2, 3, 1, 5]])

    print("colOverlapsGrid = \n%s" % colOverlapsGrid)

    poolInputsCalc = poolInputs(numCols, numRows,
                                potWidth, potHeight,
                                desiredLocalActivity,
                                minOverlap,
                                centerInhib)

    poolOut = poolInputsCalc.calculatePoolInputs(colOverlapsGrid)
    print("poolOut = \n%s" % poolOut)

