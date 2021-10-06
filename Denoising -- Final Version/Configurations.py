width = 32
height = 32
Epochs = 4
Batch = 32
iterations = 10
Epoch_disc=3
Epoch_gen=1

AEfilters = [248,128,64]
LatentDim = 3500
DiscrLayers = [128,128,128,128]
DiscrStrides = [2,2,2,2]

"""
AEfilters = [64,128,256]
LatentDim = 1500
DiscrLayers = [256,256,256,256]
DiscrStrides = [2,2,2,2]
"""

DiscrEpochs, DiscrBatch = 4,50

AEEpoch, AEBatch = 10,32