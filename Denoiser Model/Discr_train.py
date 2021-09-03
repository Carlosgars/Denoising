import numpy as np , cv2, time, tensorflow as tf
from Util import plot_img, plot_loss
from ConvAutoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from Configurations import Discr_batch_size, Discr_epochs

'''
Need to install h5py to save weights:
$ conda install 'h5py<3.0.0'
or
$ pip install 'h5py<3.0.0'
(latest version raises an error when loading weights)
'''

def trainDiscr(discriminator, train, train_labels, valid, valid_labels, test, test_labels, show_loss=True, save_weights=False):
	discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
	history = discriminator.fit(train, train_labels, Discr_batch_size, Discr_epochs, shuffle=True, validation_data=(valid, valid_labels))
	if(show_loss):
		plot_loss(history)
		scores = discriminator.evaluate(test, test_labels, verbose=2)
		print("test loss: %.5f" % (scores))

	if(save_weights):
		weights = discriminator.save_weights("DiscrWeights_localrun.h5")