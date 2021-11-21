from Configurations import *
from Noiser_MT import Dataset_Builder, Writer, joiner
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import os ,time
from ConvAutoencoder import ConvAutoencoder
from Util import plot_img, plot_images



def modelFromWeightsToImage(images,weights,height, width, depth, filters, latentDim, intensity=1):
	(encoder, decoder, autoencoder) = ConvAutoencoder.build(height, width, depth, filters, latentDim)
	autoencoder.load_weights(weights)

	denoised = autoencoder.predict(images)
	diff = denoised - images
	output = images + intensity * diff
	return output


Load_path="COCODataset/images/unlabeled2017"
Save_path="COCODataset/elaborated"
ae_weights='Weights/version2.h5'

executor = ThreadPoolExecutor(max_workers=1)
future=executor.submit(Dataset_Builder, 10, 1, 1, width, Load_path, silence=False)
print("Dataset builder main thread launched")
print("Now waiting for the dataset to complete if it hasn't still finished...")
shape, clean, noise =future.result()
executor.shutdown()

images = noise[0][0]
print(images.shape)

denoised = modelFromWeightsToImage(images,ae_weights,height, width, 3, AEfilters, LatentDim, intensity=1)
halfdenoised = modelFromWeightsToImage(images,ae_weights,height, width, 3, AEfilters, LatentDim, intensity=0.65)
twicedenoised = modelFromWeightsToImage(images,ae_weights,height, width, 3, AEfilters, LatentDim, intensity=2)

plot_img(10,images,halfdenoised,denoised)