from Noiser_MT import Dataset_Builder, Writer, joiner
from concurrent.futures import ThreadPoolExecutor
import os
from shutil import rmtree
from PIL  import Image
from Configurations import *
import skimage

Load_path="COCODataset/images/unlabeled2017"
Save_path="COCODataset/elaborated"

#OUTPUT: dictionary with index the name of the image
#        inside list of three elements:
#           - size of the original image
#           - clean 5D image 
#           - noisy 5D image




#calling the functions this way will make active wait untill the process is completed

clean, noise =Dataset_Builder(Batch, Epochs, Epoch_disc+Epoch_gen, width, Load_path, silence=True) #images_per_thread=???
print(str(clean.shape) + "   " + str(clean.nbytes//1024//1024) + "MB")
print(noise.shape)



#DO SOME MAGIC WITH Dataset

#Writer(Dataset, image_size, Save_path, silence=False)





#executing the functions on different threads allows to do something else while waiting for the threads to complete

#executor = ThreadPoolExecutor(max_workers=1)
#future=executor.submit(Dataset_Builder, batch_size, image_size, Load_path, silence=False)

############################################################################
# DO SOMETHING ELSE CONCURRENTLY WHILE WAITING FOR THE DATASET TO BE BUILT #
############################################################################

#NOW I NEED THE RESULT
#Dataset=future.result()


#DO SOME MAGIC WITH Dataset

#IF THE SOME IMAGES NEED TO BE STORED
#executor.submit(Writer, Dataset, image_size, Save_path, silence=False)





#TESTING SKITIMAGE LIBRARY FOR ERROR CALCULATION

#if os.path.exists(Save_path):
#	rmtree(Save_path)
#if not os.path.exists(Save_path):
#	os.mkdir(Save_path)
#for file in Dataset.items():
#
#	img=joiner(file[1][1],file[1][0], image_size)
#	cln=joiner(file[1][2],file[1][0], image_size)
#	tmp=information_extractor(cln,img,cln)
#	j=0
#	for d in tmp:
#		Image.fromarray(skimage.util.img_as_ubyte(d)).save(Save_path+"/"+file[0]+" "+str(j)+".png")
#		j+=1







