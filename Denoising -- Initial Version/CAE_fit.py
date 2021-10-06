import numpy as np
import cv2
import os ,time

import tensorflow as tf

from ConvAutoencoder import ConvAutoencoder
from trainAuxiliarFn import add_gaussian
from Util import plot_img
from Configurations import *
from trainAuxiliarFn import add_gaussian
from CAE_train import trainCAE


#### Paths ####

images_folder = 'CroppedBicycle'



def build_train():
	train = []
	for filename in os.listdir(images_folder):
		img = cv2.imread(os.path.join(images_folder,filename))
		assert img.shape == (32,32,3), "img %s has shape %r" % (filename, img.shape)
		train.append(img)
	return np.array(train)

train = build_train()
train = train.astype('float32') / 255.
train = np.reshape(train, (len(train), 32, 32, 3))

valid = train[500:550]
test = train[550:600]
train = train[:500]

valid_noisy = add_gaussian(valid)
test_noisy = add_gaussian(test)
train_noisy = add_gaussian(train)

(encoder, decoder, autoencoder) = ConvAutoencoder.build(32, 32, 3, AEfilters, LatentDim)

trainCAE(encoder, decoder, autoencoder, train_noisy, train, valid_noisy, valid, test_noisy, test, save_weights=True)


