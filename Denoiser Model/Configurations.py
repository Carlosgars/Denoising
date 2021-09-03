width = 32
height = 32
Epochs = 100
Batch = 256
iterations = 10
Epoch_disc=5
Epoch_gen=5

AEfilters = [32,64]
LatentDim = 128
DiscrLayers = [64,64]
DiscrStrides = [2,2]

"""
AEfilters = [64,128,256]
LatentDim = 1500
DiscrLayers = [256,256,256,256]
DiscrStrides = [2,2,2,2]
"""

# training AutoEncoder separatelly
AE_epochs = 100
AE_batch_size = 64

# training Discriminator separatelly
Discr_epochs = 50
Discr_batch_size = 64