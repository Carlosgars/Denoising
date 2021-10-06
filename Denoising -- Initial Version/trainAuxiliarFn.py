from tensorflow import GradientTape
import numpy as np
from Util import plot_img
from ConvAutoencoder import ConvAutoencoder
from Discriminator import Discriminator
from tensorflow.keras.optimizers import Adam, RMSprop


def add_gaussian(batch,noise_factor=0.1):
  '''
  Add gaussian noise to a given batch.
  '''
  noisy_batch = batch + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=batch.shape) 
  return np.clip(noisy_batch, 0., 1.).astype('float32')


def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    return np.asarray(data_shuffle)


