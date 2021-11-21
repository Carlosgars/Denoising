from Configurations import *
import tensorflow as tf
from GAN import GAN
from ConvAutoencoder import ConvAutoencoder
from Discriminator import Discriminator
from LossFn import gan_lossfn, wasserstein_loss, discr_loss


gan , discriminator, autoencoder, discriminator_optimizer, gan_optimizer = GAN.build(
	height,width,3,[],[],
	gan_lossfn,AEfilters,LatentDim,
	DiscrLayers,DiscrStrides,
	init_weights=False)

(encoder, decoder, autoencoder) = ConvAutoencoder.build(
	32, 32, 3, AEfilters, LatentDim)


models = [autoencoder,discriminator,gan]
names = ['autoencoder','discriminator','gan']

for model,name in zip(models,names):
	parameters = model.count_params()
	print(name, 'has', parameters, 'parameters.')
