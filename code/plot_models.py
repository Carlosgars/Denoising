from Configurations import *
import tensorflow as tf
from GAN import GAN
from ConvAutoencoder import ConvAutoencoder
from Discriminator import Discriminator
from LossFn import gan_lossfn, wasserstein_loss, discr_loss
from tensorflow.keras.utils import plot_model

gan , discriminator, autoencoder, discriminator_optimizer, gan_optimizer = GAN.build(
	height,width,3,[],[],gan_lossfn,AEfilters,LatentDim,
	DiscrLayers,DiscrStrides,init_weights=False)


(encoder, decoder, autoencoder) = ConvAutoencoder.build(
	32, 32, 3, AEfilters, LatentDim)

for x in ['TB','LR']:
	## Encoder
	plot_model(
		encoder, to_file='encoder_shapes'+x+'.png', show_shapes=True, show_dtype=False,
		show_layer_names=False, rankdir=x, expand_nested=False, dpi=96)
	plot_model(
		encoder, to_file='encoder'+x+'.png', show_shapes=False, show_dtype=False,
		show_layer_names=False, rankdir=x, expand_nested=False, dpi=96) 

	## Decoder
	plot_model(
		decoder, to_file='decoder_shapes'+x+'.png', show_shapes=True, show_dtype=False,
		show_layer_names=False, rankdir=x, expand_nested=False, dpi=96)
	plot_model(
		decoder, to_file='decoder'+x+'.png', show_shapes=False, show_dtype=False,
		show_layer_names=False, rankdir=x, expand_nested=False, dpi=96)


	## Autoencoder
	plot_model(
		autoencoder, to_file='autoencoder_shapes'+x+'.png', show_shapes=True, show_dtype=False,
		show_layer_names=False, rankdir=x, expand_nested=True, dpi=96)
	plot_model(
		autoencoder, to_file='autoencoder'+x+'.png', show_shapes=False, show_dtype=False,
		show_layer_names=False, rankdir=x, expand_nested=True, dpi=96)
	plot_model(
		autoencoder, to_file='autoencoder_short'+x+'.png', show_shapes=False, show_dtype=False,
		show_layer_names=False, rankdir=x, expand_nested=False, dpi=96)

	## Discriminator
	plot_model(
		discriminator, to_file='discriminator_shapes'+x+'.png', show_shapes=True, show_dtype=False,
		show_layer_names=False, rankdir=x, expand_nested=False, dpi=96)
	plot_model(
		discriminator, to_file='discriminator'+x+'.png', show_shapes=False, show_dtype=False,
		show_layer_names=False, rankdir=x, expand_nested=False, dpi=96)

	## Gan
	plot_model(
		gan, to_file='gan_shapes'+x+'.png', show_shapes=True, show_dtype=False,
		show_layer_names=False, rankdir=x, expand_nested=True, dpi=96)

	plot_model(
		gan, to_file='gan'+x+'.png', show_shapes=False, show_dtype=False,
		show_layer_names=False, rankdir=x, expand_nested=True, dpi=96)

	plot_model(
		gan, to_file='gan_short'+x+'.png', show_shapes=False, show_dtype=False,
		show_layer_names=False, rankdir=x, expand_nested=False, dpi=96)