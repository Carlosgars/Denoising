import matplotlib.pyplot as plt
import cv2


def plot_img(n, original_data, input_data, decoded_imgs):
  plt.figure(figsize=(20, 8))
  for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(original_data[i]) #reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display original
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(input_data[i]) #reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i]) #reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()

def plot_images(images):
  ''' 
  Plot input BGR images as RGB images
  
  Input: list 
    List of BGR images
  '''

  fig, ax = plt.subplots(nrows = 1, ncols = len(images), figsize = (4*len(images), 4*len(images)))
  for i, p in enumerate(images):
    ax[i].imshow(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
  plt.show()


def compute_loss(model,X,Y,loss_fn):
  '''
  returns loss for a given test set. After every epoch step, compute loss and save it in a list.
  '''
  model_output = model.predict(X)
  l = loss_fn(model_output,Y)
  return l

def plot_loss(n_epochs,loss_epochs,name):
  '''
  Take list of losses through epochs and plot it in a graph epochs/loss
  '''
  epochs = [x+1 for x in range(n_epochs)]
  plt.plot(epochs,loss_epochs,label=name)
  plt.xticks(epochs)
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.show()

def output_through_epochs(autoencoder_weight_filenames,model,clean_image,noisy_image,rows,columns):
  '''
  Take a given image (array, not path), model, list of filenames of the weights through epochs. 
  Output graphic with the output of the model for that image for every epoch. The first image is the original
  '''
  f, axarr = plt.subplots(rows,columns)  
  noise
  axarr[0].plot(clean_image)
  axarr[1].plot(noisy_image)
  i = 2
  for weights in autoencoder_weight_filenames:
    model = model.load_weights(weights)
    model_output = model.predict(noisy_image)
    axarr[i].imshow(model_output)
    axarr[i].gray()
    axarr[i].get_xaxis().set_visible(False)
    axarr[i].get_yaxis().set_visible(False)
    i += 1
  plt.show()
