import cv2
import os

'''
Crop each image on the dataset to a fixed size (height,width,depth=3) which can be fed to the network
'''

origin_repo = '/Users/cgs/Desktop/Bicycle'
destiny_repo = '/Users/cgs/Desktop/CroppedBicycle'

def load_and_crop_images_from_folder(origin,destiny,dataset_size,height,width):
	x, y = 0, 0
	h , w = height, width
	n = 0
	for filename in os.listdir(origin):
		img = cv2.imread(os.path.join(origin,filename))
		if img is not None:
			crop = img[y:y+h,x:x+w]
			path = destiny
			cv2.imwrite(os.path.join(path , 'cropped_' + filename), crop)
			n += 1
			print("Cropped " + filename + " saved successfully")
		if n > dataset_size:
			break
	return str(dataset_size) + " cropped images successfully saved"


print(load_and_crop_images_from_folder(origin_repo,destiny_repo,600,32,32))


