import numpy as np
import math
import random
from htm_calc import tf_overlap as overlap
from htm_calc import tf_inhibition as inhibition


class tfHtmLayer:
    def __init__(self, params):
        # The columns are in a 2 dimensional array columnArrayWidth by columnArrayHeight.
        self.width = params['columnArrayWidth']
        self.height = params['columnArrayHeight']
        self.cellsPerColumn = params['cellsPerColumn']
        self.centerPotSynapses = params['centerPotSynapses']
        self.potentialWidth = params['potentialWidth']
        self.potentialHeight = params['potentialHeight']
        self.minOverlap = params['minOverlap']
        self.wrapInput = params['wrapInput']
        self.inhibitionHeight = params['inhibitionHeight']
        self.inhibitionWidth = params['inhibitionWidth']
        self.desiredLocalActivity = params['desiredLocalActivity']
        self.connectPermanence = params['connectPermanence']
        self.spatialPermanenceInc = params['spatialPermanenceInc']
        self.spatialPermanenceDec = params['spatialPermanenceDec']
        self.tempPermanenceInc = params['spatialPermanenceInc']
        self.tempPermanenceDec = params['spatialPermanenceDec']
        self.tempSpatialPermanenceInc = params['tempSpatialPermanenceInc']
        self.tempSeqPermanenceInc = params['tempSeqPermanenceInc']
        self.tempDelayLength = params['tempDelayLength']
        self.activeColPermanenceDec = params['activeColPermanenceDec']
        self.permanenceInc = params['permanenceInc']
        self.permanenceDec = params['permanenceDec']
        self.numInputs = params['numInputs']
        self.inputHeight = params['inputHeight']
        self.inputWidth = params['inputWidth']

        self.overlapCalc = None
        self.inhibCalc = None
        self.setupCalculators()

    def setupCalculators(self):
        # Setup the theano calculator classes used to calculate
        # efficiently the spatial, temporal and sequence pooling.
        self.overlapCalc = overlap.OverlapCalculator(self.potentialWidth,
                                                     self.potentialHeight,
                                                     self.width,
                                                     self.height,
                                                     self.inputWidth,
                                                     self.inputHeight,
                                                     self.centerPotSynapses,
                                                     self.connectPermanence,
                                                     self.minOverlap,
                                                     self.wrapInput)

        # Get the output tensors and link them to the input tensors of the next calculators
        # This creates a continuous tensorflow graph so tensors can flow through it with having
        # to be converted back to numpy arrays.
        overlapTensor = self.overlapCalc.getOverlapTensor()
        potOverlapTensor = self.overlapCalc.getPotOverlapTensor()

        self.inhibCalc = inhibition.inhibitionCalculator(self.width, self.height,
                                                         self.inhibitionWidth,
                                                         self.inhibitionHeight,
                                                         self.desiredLocalActivity,
                                                         self.minOverlap,
                                                         self.centerPotSynapses,
                                                         overlapTensor,
                                                         potOverlapTensor)


if __name__ == '__main__':

    param = {
            'numInputs': 1,
            'inputHeight': 3,
            'inputWidth': 3,
            'columnArrayWidth': 3,
            'columnArrayHeight': 3,
            'cellsPerColumn': 3,
            'desiredLocalActivity': 2,
            'minOverlap': 2,
            'wrapInput': 0,
            'inhibitionWidth': 2,
            'inhibitionHeight': 2,
            'centerPotSynapses': 1,
            'potentialWidth': 2,
            'potentialHeight': 2,
            'spatialPermanenceInc': 0.1,
            'spatialPermanenceDec': 0.02,
            'activeColPermanenceDec': 0.02,
            'tempDelayLength': 3,
            'permanenceInc': 0.1,
            'permanenceDec': 0.02,
            'tempSpatialPermanenceInc': 0.1,
            'tempSeqPermanenceInc': 0.1,
            'connectPermanence': 0.3,
            'minThreshold': 5,
            'minScoreThreshold': 5,
            'newSynapseCount': 10,
            'maxNumSegments': 10,
            'activationThreshold': 6,
            'dutyCycleAverageLength': 1000,
            'colSynPermanence': 0.2,
            'cellSynPermanence': 0.4
        }

    numColumns = param['columnArrayWidth'] * param['columnArrayHeight']
    numPotSyn = param['potentialWidth'] * param['potentialHeight']

    # Create an array representing the permanences of colums synapses
    colSynPerm = np.random.rand(numColumns, numPotSyn)
    conColSynPerm = np.array(np.where(colSynPerm > param['connectPermanence'], 1, 0))
    #print("conColSynPerm = \n%s" % conColSynPerm)

    newInputMat = np.array(np.random.randint(2, size=(param['numInputs'], param['inputHeight'], param['inputWidth'])))
    #print("newInputMat = \n%s" % newInputMat)

    htmLayer = tfHtmLayer(param)
