from Configurations import *
from Noiser_MT import Dataset_Builder, Writer, joiner
from concurrent.futures import ThreadPoolExecutor
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
from LossFn import gan_lossfn, wasserstein_loss, discr_loss
from Util import plot_img
from trainlib import train

#### Paths ####

inic_discr_weights, inic_gen_weights = 'DiscrWeights_localrun.h5', 'AEWeights_localrun.h5'

Load_path="COCODataset/images/unlabeled2017"
Save_path="COCODataset/elaborated"
ae_weights='Weights/version2.h5'
discr_weights="Weights/disc_version3.h5"

#### Build GAN model ####

gan , discriminator, autoencoder, discriminator_optimizer, gan_optimizer = GAN.build(height,width,3,inic_discr_weights,inic_gen_weights,gan_lossfn,AEfilters,LatentDim,DiscrLayers,DiscrStrides,init_weights=False)
autoencoder.load_weights(ae_weights)
discriminator.load_weights(discr_weights)

for i in range(25):
	print(i, "iteration of Dataset building and training")
	executor = ThreadPoolExecutor(max_workers=1)
	future=executor.submit(Dataset_Builder, Batch, Epochs, Epoch_disc+Epoch_gen, width, Load_path, silence=False)
	print("Dataset builder main thread launched")

	print("Now waiting for the dataset to complete if it hasn't still finished...")
	shape, clean, noise =future.result()
	executor.shutdown()
	#### Train the model ####

	train(discriminator, autoencoder, discriminator_optimizer, gan_optimizer, clean, noise, discr_loss, gan_lossfn, debug=False)
	'''
	if i % 10 == 0:
		for n in range(1):
			plot_img(5,clean[0][0][5*n:(5*n+5)],noise[0][0][5*n:(5*n+5)],autoencoder.predict(noise[0][0][5*n:(5*n+5)]))
			plot_img(5,clean[1][1][5*n:(5*n+5)],noise[1][1][5*n:(5*n+5)],autoencoder.predict(noise[1][1][5*n:(5*n+5)]))
			plot_img(5,clean[2][2][5*n:(5*n+5)],noise[2][2][5*n:(5*n+5)],autoencoder.predict(noise[2][2][5*n:(5*n+5)]))
	'''

#autoencoder.save_weights("ae_versionXXX.h5")
#discriminator.save_weights("disc_versionXXX.h5")


for n in range(4):
	plot_img(5,clean[0][0][5*n:(5*n+5)],noise[0][0][5*n:(5*n+5)],autoencoder.predict(noise[0][0][5*n:(5*n+5)]))
	plot_img(5,clean[1][1][5*n:(5*n+5)],noise[1][1][5*n:(5*n+5)],autoencoder.predict(noise[1][1][5*n:(5*n+5)]))
	plot_img(5,clean[2][2][5*n:(5*n+5)],noise[2][2][5*n:(5*n+5)],autoencoder.predict(noise[2][2][5*n:(5*n+5)]))