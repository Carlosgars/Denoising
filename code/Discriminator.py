from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

class Discriminator:
  @staticmethod
  def build(width, height, depth,filters,strides):
    chanDim = -1
    discriminator_input = Input(shape=(height, width, depth))

    for (f,s) in zip(filters,strides):
      x = Conv2D(f, s)(discriminator_input)
      x = BatchNormalization(axis=chanDim)(x)
      x = LeakyReLU()(x)
  
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(discriminator_input, x)
    return discriminator