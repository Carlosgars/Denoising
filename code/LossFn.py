from tensorflow.keras.backend import mean
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

def wasserstein_loss(y_true, y_pred):
  return mean(y_true * y_pred)

pixel_loss = MeanSquaredError()

def discr_loss(y, prediction): 
  return BinaryCrossentropy()(y, prediction) # BinaryCrossentropy()

def gan_lossfn(loss1_args,  loss2_args, coefl1=0.3, coefl2=0.7):
  '''
  Loss function for GAN model. Takes in account both pixel loss and Wass loss,
  ie, both difference between clean and denoised image and performance of the
  Discriminator.
  '''
  # loss1_args: arguments to loss_1, as tuple.
  # loss2_args: arguments to loss_2, as tuple.
  l1_value = discr_loss(*loss1_args)
  l2_value = pixel_loss(*loss2_args)
  loss_value = coefl1 * l1_value + coefl2 * l2_value
  return loss_value

# Functions used in the models

d_loss = discr_loss
g_loss = gan_lossfn