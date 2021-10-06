import numpy as np , cv2, time, tensorflow as tf

from tensorflow import GradientTape
from GAN import GAN

from ConvAutoencoder import ConvAutoencoder
from Discriminator import Discriminator
from LossFn import gan_lossfn, wasserstein_loss
from Util import plot_img
from Configurations import *


def step_discriminator(batch, batch_size, noisy_batch, autoencoder, discriminator, discriminator_optimizer, d_loss_fn, debug=False):
    if(debug):
        print("---Begginning of Discriminator Epoch---")
    middle,start_tot = time.time(),time.time()
    generated_images = autoencoder(noisy_batch)
    if(debug):
        print("feed foward completed in: %s seconds total---" % (time.time() - middle))
    
    middle = time.time()
    combined_images = np.concatenate([generated_images, batch])
    labels = np.concatenate([np.ones((batch_size, 1)),
                           np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)
    #debug
    #print('labels for discr step')
    #print(labels)
    #print(labels.dtype)
    #debug
    
    if(debug):
        print("preparing input completed in: %s seconds total---" % (time.time() - middle))
    
    middle = time.time()
    '''
    with GradientTape() as tape:
      predictions = discriminator(combined_images)
      #debug
      #print('discriminator predictions')
      #print(predictions)
      #print(predictions.dtype)
      #debug
      d_loss = d_loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    '''
    d_loss = discriminator.train_on_batch(combined_images, labels)

    if(debug):
        print("discriminator prediction completed in: %s seconds total" % (time.time() - middle))
    
    middle = time.time()
    #discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))
    if(debug):
        print("discriminator backprop completed in: %s seconds total" % (time.time() - middle))
        print("---Epoch completed in: %s seconds total---\n" % (time.time() - start_tot))
    return d_loss


def step_autoencoder(batch, batch_size, noisy_batch, autoencoder, discriminator, gan_optimizer, g_loss_fn, debug=False):
    if(debug):
        print("---Begginning of Autoencoder Epoch---")
    middle,start_tot = time.time(),time.time()
    
    misleading_labels = np.zeros((batch_size, 1)).astype('float32')
    #debug
    #print('Misleading_labels for gen step')
    #print(misleading_labels)
    #print(misleading_labels.dtype)
    #debug
    if(debug):
        print("preparing input completed in: %s seconds total---" % (time.time() - middle))
    
    middle = time.time()
    with GradientTape() as tape:
        denoised = autoencoder(noisy_batch)
        predictions = discriminator(denoised)

        #debug
        #print('predictions of discr on mislabeled')
        #print(predictions)
        #print(predictions.dtype)
        #debug
        g_loss = g_loss_fn((misleading_labels,predictions), (denoised,batch),0.5,0.5)
    grads = tape.gradient(g_loss, autoencoder.trainable_weights)
    if(debug):
        print("discriminator prediction completed in: %s seconds total" % (time.time() - middle))
    
    middle = time.time()
    gan_optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights),experimental_aggregate_gradients=False) 
    if(debug):
        print("gan optimizer completed in: %s seconds total" % (time.time() - middle))
        print("---Epoch completed in: %s seconds total---\n" % (time.time() - start_tot))
    return g_loss


def train(discriminator, autoencoder, discriminator_optimizer, gan_optimizer, clean, noise, d_loss_fn, g_loss_fn, debug=False):
    
    for epoch in range(Epochs):
        if(debug):
            print("     ***** Begginning of a new EPOCH %i*****     " % (epoch))
        middle,start_tot = time.time(),time.time()
        for i in range(Epoch_disc):
            d_loss=step_discriminator(clean[epoch][i], Batch, noise[epoch][i], autoencoder, discriminator, discriminator_optimizer, d_loss_fn, debug)
        if(debug):
            print("--- Discriminator training completed in: %s seconds total ---\n" % (time.time() - middle))
        middle = time.time()  
        
        for i in range(Epoch_gen):
            g_loss=step_autoencoder(clean[epoch][i+Epoch_disc].astype('float32'), Batch, noise[epoch][i+Epoch_disc].astype('float32'), autoencoder, discriminator, gan_optimizer, g_loss_fn, debug)
        if(debug):
            print("--- Autoencoder training completed in: %s seconds total ---\n" % (time.time() - middle))
        
        if(debug):
            print("***** EPOCH Ended in: %s seconds total *****\n\n\n" % (time.time() - start_tot))
            
        #if epoch % 1 == 0:
        print('discriminator loss at step %s: %s' % (epoch, d_loss))
        print('adversarial loss at step %s: %s' % (epoch, g_loss))
            
    
    # plot_img(5,clean[0][0][0:5],noise[0][0][0:5],autoencoder.predict(noise[0][0][0:5]))