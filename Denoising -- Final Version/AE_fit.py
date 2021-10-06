from Configurations import *
from Noiser_MT import Dataset_Builder, Writer, joiner
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import os ,time
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from ConvAutoencoder import ConvAutoencoder
from Util import plot_img, plot_images
from tensorflow.keras.optimizers import Adam
from Configurations import AEEpoch, AEBatch

'''
Need to install h5py to save weights:
$ conda install 'h5py<3.0.0'
or
$ pip install 'h5py<3.0.0'
(latest version raises an error when loading weights)
'''

#### Paths ####

Load_path="COCODataset/images/unlabeled2017"
Save_path="COCODataset/elaborated"

for n in range(5):
	executor = ThreadPoolExecutor(max_workers=1)
	future=executor.submit(Dataset_Builder, AEBatch, AEEpoch, AEBatch, width, Load_path, silence=False)
	print("Dataset builder main thread launched")


	#### Build Discriminator ####

	(encoder, decoder, autoencoder) = ConvAutoencoder.build(32, 32, 3, AEfilters, LatentDim)
	autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

	print("Now waiting for the dataset to complete if it hasn't still finished...")
	clean, noise =future.result()
	executor.shutdown()


	for e in range(AEEpoch):
		print(e,' epoch')
		epoch_clean = clean[e]
		epoch_noise = clean[e]
		for i in range(AEBatch):
			batch_clean = epoch_clean[i]
			batch_noise = epoch_noise[i]
			print(batch_clean.shape)
			print(i,autoencoder.fit(batch_clean, batch_noise))

	'''
	for e in range(AEEpoch):
		print(e,' epoch')
		epoch_clean = clean[e]
		epoch_noise = noise[e]
		print(autoencoder.fit(epoch_clean, epoch_noise))
	'''

	'''
	Running this file will save the weights of the generator to be itilialized when building the GAN

	'''

	#autoencoder.save_weights("AEweights_localrun.h5")

	'''
	Checking output

	'''

	Test_num = 10
	tmp=autoencoder.predict(noise[0][0][0:Test_num])
	plot_img(Test_num,clean[0][0][0:Test_num],clean[0][0][0:Test_num],tmp)

