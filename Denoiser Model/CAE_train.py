import numpy as np , cv2, time, tensorflow as tf

from Util import plot_img, plot_loss
from ConvAutoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from Configurations import AE_epochs, AE_batch_size

'''
Need to install h5py to save weights:
$ conda install 'h5py<3.0.0'
or
$ pip install 'h5py<3.0.0'
(latest version raises an error when loading weights)
'''

def trainCAE(encoder, decoder, autoencoder, x_train_noisy, x_train, x_valid_noisy, x_valid, x_test_noisy, x_test, show_test=True, show_loss=True, save_weights=False):
	autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
	history = autoencoder.fit(x_train_noisy, x_train, AE_batch_size, AE_epochs, shuffle=True, validation_data=(x_valid_noisy, x_valid))
	if(show_loss):
		plot_loss(history)
		scores = autoencoder.evaluate(x_test_noisy, x_test, verbose=2)
		print("test mse: %.5f" % (scores))

	if(show_test):
		encoded_imgs = encoder.predict(x_test_noisy)
		decoded_imgs = decoder.predict(encoded_imgs)
		plot_img(10, x_test, x_test_noisy, decoded_imgs)

	if(save_weights):
		weights = autoencoder.save_weights("CAEweights_localrun.h5")

