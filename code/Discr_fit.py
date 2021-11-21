from Configurations import *
from Noiser_MT import Dataset_Builder, Writer, joiner
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import os ,time
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from Configurations import DiscrEpochs, DiscrBatch
from Discriminator import Discriminator
from Util import plot_img

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

executor = ThreadPoolExecutor(max_workers=1)
future=executor.submit(Dataset_Builder, DiscrBatch, DiscrEpochs, DiscrBatch, width, Load_path, silence=True)
print("Dataset builder main thread launched")


#### Build Discriminator ####

discriminator = Discriminator.build(32, 32, 3, DiscrLayers,DiscrStrides)
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=BinaryCrossentropy(from_logits=True))

print("Now waiting for the dataset to complete if it hasn't still finished...")
clean, noise =future.result()
executor.shutdown()

for e in range(DiscrEpochs-1):
	print(e,' epoch')
	epoch_clean = clean[e]
	epoch_noise = noise[e]
	for i in range(DiscrBatch):
		batch_clean = epoch_clean[i]
		batch_noise = epoch_noise[i]
		print(batch_clean.shape)
		combine_clean_noise = np.concatenate([batch_clean, batch_noise])
		labels = np.concatenate([np.ones((DiscrBatch, 1)),
                           np.zeros((DiscrBatch, 1))])
		print(i, 'th iteration. ', 'Train loss:', discriminator.train_on_batch(combine_clean_noise, labels))


'''
running this file will save the weights of the discriminator to be itilialized when building the GAN
'''
discriminator.save_weights("DiscrWeights_localrun.h5")


''' User last epoch from dataset to check output '''
print('Clean predictions should be 1', discriminator.predict(clean[DiscrEpochs-1][0]))
print('Noise predictions should be 0', discriminator.predict(noise[DiscrEpochs-1][0]))
