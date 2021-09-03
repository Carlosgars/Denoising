import matplotlib.pyplot as plt
import cv2


def plot_img(n, original_data, input_data, decoded_imgs):
  plt.figure(figsize=(20, 8))
  for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(original_data[i].reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display original
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(input_data[i].reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(32, 32,3))
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


## For ConvAutoencoder 
def plot_loss(history):
  plt.figure(figsize=(10,6))
  plt.plot(history.epoch,history.history['loss'])
  plt.plot(history.epoch,history.history['val_loss'])
  plt.title('loss')