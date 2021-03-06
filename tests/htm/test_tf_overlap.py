
import numpy as np
from htm_calc import tf_overlap


class test_tfOverlap:
    def setUp(self):
        '''
        The tf overlap class is tested with a range of
         * input sizes
         * potential synpase sizes
         * HTM column sizes

        '''
    def test_smallNumCol(self):
        '''
        Test the tensorflow overlap calculator when there is very few columns.
        '''
        potWidth = 2
        potHeight = 2
        centerPotSynapses = 1
        numColumnRows = 5
        numColumnCols = 5
        connectedPerm = 0.3
        minOverlap = 0
        wrapInput = 0

        numInputRows = 2
        numInputCols = 2
        numInputs = 1
        numPotSyn = potWidth * potHeight
        numColumns = numColumnRows * numColumnCols

        newInputMat = np.random.randint(2, size=(numInputs, numInputRows, numInputCols))
        print("newInputMat = \n%s" % newInputMat)
        # Create an array representing the permanences of colums synapses
        colSynPerm = np.ones((numColumns, numPotSyn))

        # Create an instance of the overlap calculation class
        overlapCalc = tf_overlap.OverlapCalculator(potWidth,
                                                   potHeight,
                                                   numColumnCols,
                                                   numColumnRows,
                                                   numInputCols,
                                                   numInputRows,
                                                   centerPotSynapses,
                                                   connectedPerm,
                                                   minOverlap,
                                                   wrapInput)

        #import ipdb; ipdb.set_trace()
        #columnPotSynPositions = overlapCalc.getPotentialSynapsePos(numInputCols, numInputRows)
        colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)
        print("colOverlaps = \n", colOverlaps)
        #print("columnPotSynPositions = \n", columnPotSynPositions)

    def test_getPotentialSynapsePos(self):
        '''
        Test the teensorflow overlap calculators getPotentialSynapsePos
        function.
        '''
        potWidth = 3
        potHeight = 3
        centerPotSynapses = 1
        numColumnRows = 3
        numColumnCols = 3
        connectedPerm = 0.3
        minOverlap = 3
        wrapInput = 0

        numInputRows = 3
        numInputCols = 3
        numInputs = 1
        numPotSyn = potWidth * potHeight
        numColumns = numColumnRows * numColumnCols

        newInputMat = np.random.randint(2, size=(numInputs, numInputRows, numInputCols))
        # Create an array representing the permanences of colums synapses
        colSynPerm = np.random.rand(numColumns, numPotSyn)

        # Create an instance of the overlap calculation class
        overlapCalc = tf_overlap.OverlapCalculator(potWidth,
                                                   potHeight,
                                                   numColumnCols,
                                                   numColumnRows,
                                                   numInputCols,
                                                   numInputRows,
                                                   centerPotSynapses,
                                                   connectedPerm,
                                                   minOverlap,
                                                   wrapInput)

        #import ipdb; ipdb.set_trace()
        columnPotSynPositions = overlapCalc.getPotentialSynapsePos(numInputCols, numInputRows)
        colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)

        print("columnPotSynPositions = \n", columnPotSynPositions)

        result = (np.array([[-1., -1., -1., -1., -0.,  0., -1.,  1.,  1.],
                           [-1., -1., -1., -0.,  0.,  0.,  1.,  1.,  1.],
                           [-1., -1., -1.,  0.,  0., -1.,  1.,  1., -1.],
                           [-1., -0.,  0., -1.,  1.,  1., -1.,  2.,  2.],
                           [-0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  2.],
                           [0.,  0., -1.,  1.,  1., -1.,  2.,  2., -1.],
                           [-1.,  1.,  1., -1.,  2.,  2., -1., -1., -1.],
                           [1.,  1.,  1.,  2.,  2.,  2., -1., -1., -1.],
                           [1.,  1., -1.,  2.,  2., -1., -1., -1., -1.]]),
                  np.array([[2., 2., 2., 2., 0., 1., 2., 0., 1.],
                           [2., 2., 2., 0., 1., 2., 0., 1., 2.],
                           [2., 2., 2., 1., 2., 2., 1., 2., 2.],
                           [2., 0., 1., 2., 0., 1., 2., 0., 1.],
                           [0., 1., 2., 0., 1., 2., 0., 1., 2.],
                           [1., 2., 2., 1., 2., 2., 1., 2., 2.],
                           [2., 0., 1., 2., 0., 1., 2., 2., 2.],
                           [0., 1., 2., 0., 1., 2., 2., 2., 2.],
                           [1., 2., 2., 1., 2., 2., 2., 2., 2.]]))

        np.array_equal(columnPotSynPositions, result)

    def test_inputSizes(self):
        '''
        Test the tensorflow overlap calculator with a range of input sizes
        '''
        potWidth = 4
        potHeight = 4
        centerPotSynapses = 1
        numColumnRows = 7
        numColumnCols = 5
        connectedPerm = 0.3
        minOverlap = 3
        wrapInput = 0
        numPotSyn = potWidth * potHeight
        numColumns = numColumnRows * numColumnCols
        numInputs = 1

        # Create an array representing the permanences of colums synapses
        colSynPerm = np.random.rand(numColumns, numPotSyn)

        for i in range(4, 100, 3):
            numInputRows = i
            for j in range(4, 100, 7):
                numInputCols = j
                print "NEW TEST ROUND"
                print "numInputRows, numInputCols = %s, %s " % (numInputRows, numInputCols)
                newInputMat = np.random.randint(2, size=(numInputs, numInputRows, numInputCols))
                # Create an instance of the overlap calculation class
                overlapCalc = tf_overlap.OverlapCalculator(potWidth,
                                                           potHeight,
                                                           numColumnCols,
                                                           numColumnRows,
                                                           numInputCols,
                                                           numInputRows,
                                                           centerPotSynapses,
                                                           connectedPerm,
                                                           minOverlap,
                                                           wrapInput,
                                                           numInputs
                                                           )

                # Return both the overlap values and the inputs from
                # the potential synapses to all columns.
                colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)

                columnPotSynPositions = overlapCalc.getPotentialSynapsePos(numInputCols, numInputRows)

                assert len(colOverlaps) == numColumns

    def test_minOverlap(self):
        '''
        Test the tf overlap calculator with a case where their is no
        columns with an overlap value larger then the min overlap value.
        '''
        potWidth = 2
        potHeight = 2
        centerPotSynapses = 1
        numColumnRows = 4
        numColumnCols = 4
        connectedPerm = 0.3
        minOverlap = 3
        wrapInput = 0

        colSynPerm = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])

        newInputMat = np.array([[[1, 1, 1, 1],
                                [0, 0, 0, 0],
                                [1, 1, 1, 1],
                                [0, 0, 0, 0]]])

        numInputCols = 4
        numInputRows = 4

        # Create an instance of the overlap calculation class
        overlapCalc = tf_overlap.OverlapCalculator(potWidth,
                                                   potHeight,
                                                   numColumnCols,
                                                   numColumnRows,
                                                   numInputCols,
                                                   numInputRows,
                                                   centerPotSynapses,
                                                   connectedPerm,
                                                   minOverlap,
                                                   wrapInput)

        # Return both the overlap values and the inputs from
        # the potential synapses to all columns.
        colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)

        #import ipdb; ipdb.set_trace()
        assert np.sum(colOverlaps) == 0

    def test_uncenteredCase1(self):
        '''
        Test the tf overlap calculator with a case where
        each column calculates the overlap with that columns
        potential synpases begining from the top right. The
        potential synpases are not cenetered around the column.
        '''
        potWidth = 2
        potHeight = 2
        centerPotSynapses = 0
        numColumnRows = 5
        numColumnCols = 4
        connectedPerm = 0.3
        minOverlap = 3
        wrapInput = 0

         # The below colsynPerm needs to have potWidth * potHeight number of columns
         # and needs to have numColumnCols * numColumnRows number of rows.
        colSynPerm = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])

        # Needs to have numColumnCols number of columns and
        # numColumnRows number of rows for it to be valid.
        newInputMat = np.array([[[1, 1, 1, 1],
                                [0, 0, 0, 0],
                                [1, 1, 1, 1],
                                [0, 0, 0, 0]]])

        numInputCols = 4
        numInputRows = 4

        # Create an instance of the overlap calculation class
        overlapCalc = tf_overlap.OverlapCalculator(potWidth,
                                                   potHeight,
                                                   numColumnCols,
                                                   numColumnRows,
                                                   numInputCols,
                                                   numInputRows,
                                                   centerPotSynapses,
                                                   connectedPerm,
                                                   minOverlap,
                                                   wrapInput)

        # Return both the overlap values and the inputs from
        # the potential synapses to all columns.
        colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)


        #import ipdb; ipdb.set_trace()
        assert np.sum(colOverlaps) == 0

    def test_uncenteredInputSizes(self):
        potWidth = 4
        potHeight = 5
        centerPotSynapses = 0
        numColumnRows = 7
        numColumnCols = 5
        connectedPerm = 0.3
        minOverlap = 3
        numPotSyn = potWidth * potHeight
        numColumns = numColumnRows * numColumnCols
        wrapInput = 0
        numInputs = 1

        # Create an array representing the permanences of colums synapses
        colSynPerm = np.random.rand(numColumns, numPotSyn)

        for i in range(4, 100, 3):
            numInputRows = i
            for j in range(4, 100, 7):
                numInputCols = j
                print "NEW TEST ROUND"
                print "numInputRows, numInputCols = %s, %s " % (numInputRows, numInputCols)
                newInputMat = np.random.randint(2, size=(numInputs, numInputRows, numInputCols))
                # Create an instance of the overlap calculation class
                overlapCalc = tf_overlap.OverlapCalculator(potWidth,
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

                columnPotSynPositions = overlapCalc.getPotentialSynapsePos(numInputCols, numInputRows)

                # Return both the overlap values and the inputs from
                # the potential synapses to all columns.
                colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)

                assert len(colOverlaps) == numColumns

    def test_smallPotSizes(self):
        '''
        Test the tf overlap calculator with a specific case.
        Also test the overlap calculator remove small overlap values
        with an edge case.
        '''
        potWidth = 2
        potHeight = 2
        centerPotSynapses = 0
        numColumnRows = 5
        numColumnCols = 4
        connectedPerm = 0.3
        minOverlap = 2
        wrapInput = 0

        # The below colsynPerm needs to have potWidth * potHeight number of columns
        # and needs to have numColumnCols * numColumnRows number of rows.
        colSynPerm = np.array([[0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31],
                               [0.31, 0.31, 0.31, 0.31]])

        # Needs to have numColumnCols number of columns and
        # numColumnRows number of rows for it to be valid.
        newInputMat = np.array([[[0, 0, 1, 0],
                                [0, 0, 1, 0],
                                [0, 0, 1, 0],
                                [0, 0, 1, 0]]])

        numInputCols = 4
        numInputRows = 4

        # Create an instance of the overlap calculation class
        overlapCalc = tf_overlap.OverlapCalculator(potWidth,
                                                   potHeight,
                                                   numColumnCols,
                                                   numColumnRows,
                                                   numInputCols,
                                                   numInputRows,
                                                   centerPotSynapses,
                                                   connectedPerm,
                                                   minOverlap,
                                                   wrapInput)

        # Return both the overlap values and the inputs from
        # the potential synapses to all columns.
        colOverlaps, colPotInputs = overlapCalc.calculateOverlap(colSynPerm, newInputMat)

        print "colOverlaps = ", colOverlaps
        #import ipdb; ipdb.set_trace()
        # remove the tie breaker values from the overlap scores.
        colOverlaps = np.floor(colOverlaps)
        result = [0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert np.array_equal(colOverlaps, result)


