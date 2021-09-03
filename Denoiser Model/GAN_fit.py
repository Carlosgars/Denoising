import numpy as np
import cv2
import os ,time

import tensorflow as tf
from tensorflow.keras.backend import mean
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow import GradientTape
from GAN import GAN

from ConvAutoencoder import ConvAutoencoder
from Discriminator import Discriminator
from trainAuxiliarFn import next_batch, add_gaussian
from LossFn import gan_lossfn, wasserstein_loss
from Util import plot_img
from Configurations import *

from trainlib import train

'''
Need to install h5py to load weights:
$ conda install 'h5py<3.0.0'
or
$ pip install 'h5py<3.0.0'
(latest version raises an error when loading weights)
'''

#### Paths ####

discr_weights, gen_weights = 'DiscrWeights_localrun.h5', 'CAEweights_localrun.h5'
images_folder = 'CroppedBicycle'


#### Build image array for training ####

def build_x_train():
	x_train = []
	for filename in os.listdir(images_folder):
		img = cv2.imread(os.path.join(images_folder,filename))
		assert img.shape == (32,32,3), "img %s has shape %r" % (filename, img.shape)
		x_train.append(img)
	return np.array(x_train)

x_train = build_x_train()
x_train = x_train.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))


#### Build GAN model ####

gan , discriminator, autoencoder, discriminator_optimizer, gan_optimizer = GAN.build(32,32,3,discr_weights,gen_weights,gan_lossfn,AEfilters,LatentDim,DiscrLayers,DiscrStrides, init_weights=False)


#### Train the model ####

train(discriminator, autoencoder, discriminator_optimizer, gan_optimizer, x_train, iterations, Batch, wasserstein_loss, gan_lossfn, False)

