import numpy as np , cv2, time, tensorflow as tf

from tensorflow import GradientTape
from GAN import GAN

from ConvAutoencoder import ConvAutoencoder
from Discriminator import Discriminator
from trainAuxiliarFn import next_batch, add_gaussian
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
    labels += 0.03 * np.random.random(labels.shape)
    if(debug):
        print("preparing input completed in: %s seconds total---" % (time.time() - middle))
    
    middle = time.time()
    with GradientTape() as tape:
      predictions = discriminator(combined_images)
      d_loss = d_loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    if(debug):
        print("discriminator prediction completed in: %s seconds total" % (time.time() - middle))
    
    middle = time.time()
    discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))
    if(debug):
        print("discriminator backprop completed in: %s seconds total" % (time.time() - middle))
        print("---Epoch completed in: %s seconds total---\n" % (time.time() - start_tot))
    return d_loss


def step_autoencoder(batch, batch_size, noisy_batch, autoencoder, discriminator, gan_optimizer, g_loss_fn, debug=False):
    if(debug):
        print("---Begginning of Autoencoder Epoch---")
    middle,start_tot = time.time(),time.time()
    
    misleading_labels = np.zeros((batch_size, 1))
    if(debug):
        print("preparing input completed in: %s seconds total---" % (time.time() - middle))
    
    middle = time.time()
    with GradientTape() as tape:
        denoised = autoencoder(noisy_batch)
        predictions = discriminator(denoised)
        g_loss = g_loss_fn((misleading_labels,predictions), (denoised,batch),0.4,0.6)
    grads = tape.gradient(g_loss, autoencoder.trainable_weights)
    if(debug):
        print("discriminator prediction completed in: %s seconds total" % (time.time() - middle))
    
    middle = time.time()
    gan_optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights),experimental_aggregate_gradients=False) 
    if(debug):
        print("gan optimizer completed in: %s seconds total" % (time.time() - middle))
        print("---Epoch completed in: %s seconds total---\n" % (time.time() - start_tot))
    return g_loss


def train(discriminator, autoencoder, discriminator_optimizer, gan_optimizer, x_train, iterations, batch_size, d_loss_fn, g_loss_fn, debug=False):
    
    for epoch in range(Epochs):
        if(debug):
            print("     ***** Begginning of a new EPOCH %i*****     " % (epoch))
        middle,start_tot = time.time(),time.time()
        batch= next_batch(batch_size*(Epoch_disc + Epoch_gen), x_train)
        noisy_batch = add_gaussian(batch)
        batch=np.reshape(batch,((Epoch_disc+Epoch_gen),Batch,height,width,3))
        noisy_batch=np.reshape(noisy_batch,((Epoch_disc+Epoch_gen),Batch,height,width,3))
        if(debug):
            print("--- dataset preparation completed in: %s seconds total ---\n" % (time.time() - middle))
        middle = time.time()
        
        for i in range(Epoch_disc):
            d_loss=step_discriminator(batch[i], Batch, noisy_batch[i], autoencoder, discriminator, discriminator_optimizer, d_loss_fn, debug)
        if(debug):
            print("--- Discriminator training completed in: %s seconds total ---\n" % (time.time() - middle))
        middle = time.time()  
        
        for i in range(Epoch_gen):
            g_loss=step_autoencoder(batch[i+Epoch_disc], Batch, noisy_batch[i+Epoch_disc], autoencoder, discriminator, gan_optimizer, g_loss_fn, debug)
        if(debug):
            print("--- Autoencoder training completed in: %s seconds total ---\n" % (time.time() - middle))
        
        if(debug):
            print("***** EPOCH Ended in: %s seconds total *****\n\n\n" % (time.time() - start_tot))
            
        if epoch % 10 == 0:
            print('discriminator loss at step %s: %s' % (epoch, d_loss))
            print('adversarial loss at step %s: %s' % (epoch, g_loss))
            
    noisy = add_gaussian(x_train[0:5])
    plot_img(5,x_train[0:5],noisy,autoencoder.predict(noisy))