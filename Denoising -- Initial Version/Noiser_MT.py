import skimage, os, time, random, queue
from skimage import io, filters
from skimage.util import compare_images
from numpy import ndarray, asarray, array, reshape, zeros
from PIL  import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from shutil import rmtree





def information_extractor(clean, noisy, denoise):
	difference=compare_images(clean, denoise)
	noise=compare_images(clean, noisy)
	detail_original3=compare_images(clean, filters.gaussian(clean,2))
	detail_original2=compare_images(clean, filters.gaussian(clean,1))
	tmp=compare_images(detail_original3, detail_original2, "blend")
	detail_original1=compare_images(clean, filters.gaussian(clean,0.5))
	detail_original=tmp=compare_images(tmp, detail_original1, "blend")
			#sum this 3 images to get all the information possible about detail
			#decide how to give points based on the level of detail. maybe just sum the images
	detail_denoise3=compare_images(clean, filters.gaussian(denoise,2))
	detail_denoise2=compare_images(clean, filters.gaussian(denoise,1))
	tmp=compare_images(detail_denoise3, detail_denoise2, "blend")
	detail_denoise1=compare_images(clean, filters.gaussian(denoise,0.5))
	detail_denoise=tmp=compare_images(tmp, detail_denoise1, "blend")
	detail_lost=compare_images(detail_original, detail_denoise)
	
	return [difference, noise, detail_original1, detail_original2, detail_original3, detail_denoise1, detail_denoise2, detail_denoise3, detail_denoise, detail_original, detail_lost]




def loader(q, dir_open="/preprocess", numb, batch, epoch, miniEpoch, size=64, hide_progress_bar=True):
	#dir_open=directory where the files are located, 
	#total= total number of images that will be processed, 
	#chunk_size= size of the single chunk dedicated to a single thread
	#print(type(os.listdir(dir_open)))
	if not os.path.exists(dir_open):
		print("Error! Directory non existent: " + dir_open)
		return False
	files=random.sample(os.listdir(dir_open), numb) #start with an aprox number of images
	if(not len(files)):
		print("Error: no file found at:'" + dir_open + "'")
		return False
	start = time.time()
	map=[]
	i=0
	#if len(files)<total:
	#	total=len(files)
	#chunk_number=int(total/chunk_size + (1 if (total%chunk_size) else 0))
    
	for chunk in tqdm(range(epoch),desc="Loading", position=0, disable=hide_progress_bar):
        chunksCount=0
		#for file in files[chunk*chunk_size:(1+chunk)*chunk_size]:
			#if(files.index(file)<total):
        while(chunksCount<batch*miniEpoch):
            if(i==numb):  #if the numb of images opened are not enough, reload the list of files to upload another numb of images
                files=random.sample(os.listdir(dir_open), numb)
                i=0
            img=Image.open(dir_open+"/"+file[i]
            map.append(img))
            chunksCount+=int(img.shape[0]/size + (1 if (img.shape[0]%size) else 0)) * int(img.shape[1]/size + (1 if (img.shape[1]%size) else 0))
            i+=1
		q.put(map) #every list will be an epoch of images that will be sent to a different thread
		map=[]
	if not hide_progress_bar:
		time.sleep(0.01)
		tqdm.write("--- load completed in: %s seconds ---" % (time.time() - start) )
	return
	
	
def splitter(image, size):
	#image=ndarray image
	#shape=image.shape tuple
	#size=height and width of the square chunk 
	#return=list of chunks of the mosaic original image
	shape=image.shape()
	if(shape[0]<size or shape[1]<size):
		#print("image too small")
		#TODO: implement resize to expand image
		x=(size if (shape[1]<size) else shape[1])
		y=(size if (shape[0]<size) else shape[0])
		new_image=zeros([y,x,3])
		for Y in range(shape[0]): #shapeY instead of y because we need to transfer only the smaller image into the new bigger image. 
			for X in range(shape[1]): #shapeX instead of x because we need to transfer only the smaller image into the new bigger image. 
				new_image[int(Y+(y-shape[0])/2) , int(X+(x-shape[1])/2)]=image[Y][X]
		image=new_image
		shape=[y,x]
	
	#get number of chunks for each axis
	Y_quantity=int(shape[0]/size + (1 if (shape[0]%size) else 0))
	X_quantity=int(shape[1]/size + (1 if (shape[1]%size) else 0))
	
	#list where the chunks will be saved
	chunks=[]
	
	#iterate
	for Y in range(Y_quantity):
		for X in range(X_quantity):
		
			#calculating the edges offset of the chunk Y*X_quantity+X th
			if (not shape[0]%size and not shape[1]%size):
				#perfect fit case skip everything, but how many times this case will happen?
				#extra useless if to compute maybe?
				tupleX= [X*size,(X+1)*size]
				tupleY= [Y*size,(Y+1)*size]
			
			elif (X<X_quantity-1 and Y<Y_quantity-1):
				#base condition inside the limits
				tupleX= [X*size,(X+1)*size]
				tupleY= [Y*size,(Y+1)*size]
				#debug line to get coordinates of every chunk 
				#print(str(X_quantity*Y+X)+" normal    "+ str(Y*size)+":"+ str((Y+1)*size)+" , "+ str(X*size)+":"+ str((X+1)*size))
				
			else:#if division has reminder it doesn't fit properly and last chunk on the edge
				#search for not perfect fitting in the last row / column and replace the normal chunk edges
				#with the edges from the image_dimension to image_dimension-size
				tupleY= [shape[0]-size,shape[0]] if (shape[0]%size and Y==Y_quantity-1) else [Y*size,(Y+1)*size]
				tupleX= [shape[1]-size,shape[1]] if (shape[1]%size and X==X_quantity-1) else [X*size,(X+1)*size]
				#debug line to get coordinates of every chunk 
				#print(str(X_quantity*Y+X)+" special    "+ str(tupleY[0])+":"+ str(tupleY[1]) +" , "+ str(tupleX[0])+":"+ str(tupleX[1]))
				
			#once we have the 
			#check numpy.ndarray.resize
			chunks.append(image[tupleY[0]:tupleY[1],tupleX[0]:tupleX[1]])
		chunks.append(tmp)
	chunks=array(chunks)
	return chunks
		#normal split: split from base+offset to base+offset+size_chunk
		#special split: check if last chunk is full conteined, else the chunk will be sized [Ysize-chunk_size : Ysize],[Xsize-chunk_size : Xsize]
	
def joiner(image, shape, size):
	#image=ndarray image
	#shape=image.shape tuple
	#size=height and width of the square chunk 
	#return=list of chunks of the mosaic original image
	tupleSize=image.shape
	sizex=(size if (shape[1]<size) else shape[1])
	sizey=(size if (shape[0]<size) else shape[0])
	new_image=zeros([sizey, sizex, 3])
	
	for K in range(tupleSize[0]):
		for J in range(tupleSize[1]):
			for Y in range(tupleSize[2]):
				for X in range(tupleSize[3]):
					#it's filling time
					y=K*tupleSize[2]+Y
					x=J*tupleSize[3]+X
					if (sizey%size and K==tupleSize[0]-1):
						#print("last K  "+str(K)+"  "+str(J)+"  "+str(Y)+" "+str(X))
						if Y<size-(sizey%size):
							#print("Y= "+str(Y))
							#print(str(shape[0])+" Mod "+str(size)+" = "+str(shape[0]%size))
							continue
						y-=size-(sizey%size)
					if (sizex%size and J==tupleSize[1]-1):
						#print("last J  "+str(K)+"  "+str(J)+"  "+str(Y)+" "+str(X))
						if X<size-(sizex%size):
							#print("X= "+str(X))
							#print(str(shape[0])+" Mod "+str(size)+" = "+str(shape[0]%size))
							continue
						x-=size-(sizex%size)
					#print("Y="+str(K*tupleSize[2]+Y)+ "X="+str(J*tupleSize[3]+X))
					new_image[y,x]  =  image[K*tupleSize[1]+J,Y,X]
					
					
	#now calculate the min edges to cut:
	if(shape[0]<size or shape[1]<size):
		startY=int((new_image.shape[0]-shape[0])/2) if (shape[0]<size) else 0
		startX=int((new_image.shape[1]-shape[1])/2) if (shape[1]<size) else 0
		new_image=new_image[startY:shape[0]+startY , startX:shape[1]+startX]
	
	return new_image
	

	
def noiser(map, batch, miniEpoch, size, num, d, out, silence=True):
	#map{}= dictionary of the loaded images, 
	#num= identifier of the thread,
	#d= 
    clean=np.empty(1,size,size,3)
    noise=np.empty(1,size,size,3)
	for file in map:
		image=asarray(map[file])
		image=skimage.util.img_as_float(image)
		#print(image)
        
		#map[file]=[] #empty content
		shape=image.shape
		#map[file].append(shape)
		#print(shape)
		#adding the chunked clean image
		clean.append(splitter(image, size))
		#  twick this parameters to change the range of randomness of the different noise filters
		#  NOTE: you can't set max range as 1 or even too high because there is a limit on how much 
		#  information the NN can retrieve from the non noise pixels
		_mean_min=0      		# float, 0     gaussian, speckle 
		_mean_max=0.155
		_var_min=0				# float, 0.01  gaussian, speckle   Note: variance = (standard deviation) ** 2
		_var_max=0.05
		_amount_min=0.0001				# float, 0.05  salt, pepper, s&p
		_amount_max=0.115
		_salt_vs_pepper_min=0.03		# float, 0.5   s&p
		_salt_vs_pepper_max=0.999
		
		
		tmp=random.randint(0,10)  
		#map[file].append(tmp)      activate if it is ment to save the files with the name of the filter
		if(tmp==0 or tmp==8):
			image=skimage.util.random_noise(image, mean=random.uniform(_mean_min,_mean_max), var=random.uniform(_var_min,_var_max))
		elif(tmp==1 or tmp==9):
			image=skimage.util.random_noise(image,"poisson")
		elif(tmp==2):
			image=skimage.util.random_noise(image,"salt", amount=random.uniform(_amount_min,_amount_max))
		elif(tmp==3):
			image=skimage.util.random_noise(image,"s&p", amount=random.uniform(_amount_min,_amount_max), salt_vs_pepper=random.uniform(_salt_vs_pepper_min,_salt_vs_pepper_max))	
		elif(tmp==4 or tmp== 10):
			image=skimage.util.random_noise(image,"speckle",mean=random.uniform(_mean_min,_mean_max), var=random.uniform(_var_min,_var_max))
		elif(tmp==5):
			image=skimage.util.random_noise(skimage.util.random_noise(image, mean=random.uniform(_mean_min,_mean_max), var=random.uniform(_var_min,_var_max)), "salt", amount=random.uniform(_amount_min,_amount_max))	
		elif(tmp==6):
			image=skimage.util.random_noise(skimage.util.random_noise(image,"poisson"),"salt", amount=random.uniform(_amount_min,_amount_max))
		elif(tmp==7):
			image=skimage.util.random_noise(skimage.util.random_noise(image,"speckle", mean=random.uniform(_mean_min,_mean_max), var=random.uniform(_var_min,_var_max)),"salt", amount=random.uniform(_amount_min,_amount_max))
	
		#adding the second chunked noisy image
		noise.append(splitter(image, size))
		if silence:
			out[file]=map[file]
    
    if(clean.size[3]>batch*miniEpoch)
        clean=clean[0:batch*miniEpoch]  #cut the excess out
    clean.reshape(miniEpoch,batch,size,size,3)
    if(noisy.size[3]>batch*miniEpoch)
        noisy=clean[0:batch*miniEpoch]  #cut the cexcess out
    noisy.reshape(miniEpoch,batch,size,size,3)
    
	if silence:
		return
	else:
		d.put(clean,noisy)
	return
	
	
def cache(d, out, num_chunks, hide_progress_bar=True):
	#this is only to implement the progress bar on the threads
	time.sleep(0.1)
	start = time.time()
	for _ in tqdm(range(num_chunks), desc="Workers progress", position=1, disable=hide_progress_bar):
		map=d.get(True)
		for tmp in map:
			out[tmp]=map[tmp]
	tqdm.write("--- elaboration completed in: %s seconds total---" % (time.time() - start))
	return


def Writer(input, size, dir_save="/Noise", silence=True):
	#dir_save= directory where the file will be saved, 
	#input= dictionary of the loaded images,
	#size= size of the chunk
	
	if os.path.exists(dir_save):
		rmtree(dir_save)
	if not os.path.exists(dir_save):
		os.mkdir(dir_save)
	start = time.time()
	for file in tqdm(input.items(), desc="Storing progress", disable=silence):
	
		img=joiner(file[1][1],file[1][0], size)
		#io.imshow(skimage.util.img_as_ubyte(img))
		#time.sleep(10)
		#print("caccola")
		Image.fromarray(skimage.util.img_as_ubyte(img)).save(dir_save+"/"+file[0]+" "+".png")

	if not silence:
		tqdm.write("--- writing back completed in: %s seconds total---" % (time.time() - start))
	return
	

	
def Dataset_Builder(number=10000, size=256, path="/Clear", silence=True,  images_per_thread=30):
	# number 	 =  n. of images to process 		100 default
	# size   	 =  size of image in pixel (N x N)		256 default
	# path 		 =  directory where the clear images are located		/Clear folder default
	# silence    =  silence the output. True print only the total time required. False print progress bar, for testing purpose		True default
	# save		 =  save the noisy images on the local directory		False default
	# save_path  =  IF save=True the images will be saved in the specified directory 		/Noise folder default
	# images_per_thread =  number of images per Thread, for load balancing and tuning purpose		20 default
	
	#print(os.cpu_count())
	print("start building batch dataset")
	num_chunks=int(number/images_per_thread + (1 if (number%images_per_thread) else 0))
	
	q=queue.Queue(num_chunks)
	d=queue.Queue(num_chunks)
	out={}
	
	start_tot = time.time()
	random.seed(time.time())
	
	executor = ThreadPoolExecutor(max_workers=2)
	executor.submit(loader, q, path, number, batch, epoch, miniEpoch, size=64, silence)
	if not silence:
		executor.submit(cache, d, out, num_chunks, silence)
	cores = ThreadPoolExecutor(max_workers=os.cpu_count())
	for num in range(num_chunks):
		data=q.get(True)
		cores.submit(noiser, data, size, num, d, out, silence)
		#print(cores.submit(noiser, data, size, num, d).result())
	cores.shutdown()
	#print(executor.submit(writer, s, size, save_path, number, images_per_thread, silence).result())
	executor.shutdown()
	if silence:
		print("--- dataset generation completed in: %s seconds total---" % (time.time() - start_tot))
		
	return out


