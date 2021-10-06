from ConvAutoencoder import ConvAutoencoder
from Discriminator import Discriminator

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


class GAN:
  @staticmethod
  def build(width, height, depth, discr_init, gen_init,gan_loss,gen_filters,gen_latentDim,d_filters,d_strides,init_weights=False):
    # Building Discriminator
    discriminator = Discriminator.build(width,height, depth,d_filters,d_strides)
    discriminator_optimizer = Adam(lr=0.003)
    discriminator.trainable = False
    discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
    if(init_weights):
      discriminator.load_weights(discr_init)


    # Building Generator
    (encoder, decoder, autoencoder) = ConvAutoencoder.build(width,height, depth, gen_filters, gen_latentDim)
    ae_opt = Adam(lr=1e-4)
    if(init_weights):
      autoencoder.load_weights(gen_init)


    gan_input = Input(shape=(width, height, depth))
    gan_output = discriminator(autoencoder(gan_input))
    gan = Model(gan_input, gan_output)
    gan_optimizer = RMSprop(learning_rate=0.0005, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=gan_optimizer, loss=gan_loss)
    return gan, discriminator, autoencoder, discriminator_optimizer, gan_optimizer