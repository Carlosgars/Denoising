
import time

import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import Util

from ConvAutoencoder import ConvAutoencoder

#############################################

#mixed_precision.set_global_policy('mixed_float16')

#############################################

#(x_train, _), (x_test, _) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
#
#x_valid, x_train = x_train[:20000], x_train[20000:] 
#
#x_train = x_train.astype('float32') / 255.
#x_valid = x_valid.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#
#x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
#x_valid = np.reshape(x_valid, (len(x_valid), 32, 32, 3))
#x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))
#
#noise_factor = 0.1
#x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
#x_valid_noisy = x_valid + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_valid.shape) 
#x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
#
#x_train_noisy = np.clip(x_train_noisy, 0., 1.).astype('float32')
#x_valid_noisy = np.clip(x_valid_noisy, 0., 1.).astype('float32')
#x_test_noisy = np.clip(x_test_noisy, 0., 1.).astype('float32')
#
#img = cv2.cvtColor(x_train[0], cv2.COLOR_BGR2RGB)
#noisy_img = cv2.cvtColor(x_train_noisy[0], cv2.COLOR_BGR2RGB)
#plt.imshow(img)
#plt.imshow(noisy_img)



EPOCHS = 50
BS = 64

from tensorflow.keras.optimizers import Adam
opt = Adam(lr=1e-3)

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
  # Everything that creates variables should be under the strategy scope.
  # In general this is only model construction & `compile()`.
  (encoder, decoder, autoencoder) = ConvAutoencoder.build(32, 32, 3,(32,64),900)
  autoencoder.compile(loss="mse", optimizer=opt, metrics=["accuracy"])

# Train the model on all available devices.
#model.fit(train_dataset, validation_data=val_dataset, ...)

# Test the model on all available devices.
#model.evaluate(test_dataset)




print(autoencoder.summary())
time.sleep(10)

# train the convolutional autoencoder
#H = autoencoder.fit(
#	x_train_noisy, x_train,
#	validation_data=(x_valid_noisy, x_valid),
#	epochs=EPOCHS,
#	batch_size=BS)
#
#N = np.arange(0, EPOCHS)
#plt.figure()
#plt.plot(N, H.history["loss"], label="train_loss")
#plt.plot(N, H.history["val_loss"], label="val_loss")
#plt.title("Training Loss")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss")
#plt.legend(loc="lower left")


#decoded = autoencoder.predict(x_test_noisy)
#Util.plot_img(10,x_test,x_test_noisy,decoded)


#decoded = autoencoder.predict(x_test_noisy)
#Util.plot_images([x_test[0],decoded[0]])




