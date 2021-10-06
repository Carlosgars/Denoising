import skimage, os, time, random, queue, numpy as np, traceback
from skimage import io, filters
from skimage.util import compare_images
from numpy import ndarray, asarray, array, reshape, zeros, empty
from PIL  import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from shutil import rmtree
from sklearn.utils import shuffle


#  twick this parameters to change the range of randomness of the different noise filters
#  NOTE: you can't set max range as 1 or even too high because there is a limit on how much 
#  information the NN can retrieve from the non noise pixels
_mean_min=0      		# float, 0     gaussian, speckle 
_mean_max=0.075
_var_min=0				# float, 0.01  gaussian, speckle   Note: variance = (standard deviation) ** 2
_var_max=0.03
_amount_min=0.0001				# float, 0.05  salt, pepper, s&p
_amount_max=0.035
_salt_vs_pepper_min=0.03		# float, 0.5   s&p
_salt_vs_pepper_max=0.999



def loader_test(q, numb, dir_open="/preprocess", size=64, hide_progress_bar=True):
    #dir_open=directory where the files are located, 
    #total= total number of images that will be processed, 
    #chunk_size= size of the single chunk dedicated to a single thread
    if not os.path.exists(dir_open):
        print("Error! Directory non existent: " + dir_open)
        return False
    files=random.sample(os.listdir(dir_open), numb)#int(numb*2)) #start with an aprox number of images
    if(not len(files)):
        print("Error: no file found at:'" + dir_open + "'")
        return False
    #if(len(files)<int(numb*2)):
    #    print("Error: not enough files avaliable at:'" + dir_open + "'")
    #    return False
    start = time.time()
    map=[]
    i=0
    
    while(i<numb):
        img=Image.open(dir_open+"/"+files[i])
        if img.mode == "RGB":
            map.append(img)
            i+=1
    q.put(map) #every list will be an epoch of images that will be sent to a different thread
    
    if not hide_progress_bar:
        tqdm.write("--- load completed in: %s seconds ---" % (time.time() - start) )
    return
    
    

def loader(q, numb, batch, epoch, miniEpoch, dir_open="/preprocess", size=64, hide_progress_bar=True):
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
        while(chunksCount<(batch*miniEpoch*15)):
            #print(chunksCount)
            if(i==numb):  #if the numb of images opened are not enough, reload the list of files to upload another numb of images
                #print("Cover me! I'm reloading!")
                files=random.sample(os.listdir(dir_open), numb)
                i=0
            img=Image.open(dir_open+"/"+files[i])
            if img.mode == "RGB":
                map.append(img)
                w,h=img.size
                chunksCount+=int(h/size + (1 if (h%size) else 0)) * int(w/size + (1 if (w%size) else 0))
            i+=1
        q.put(map) #every list will be an epoch of images that will be sent to a different thread
        map=[]
    if not hide_progress_bar:
        #time.sleep(0.01)
        tqdm.write("--- load completed in: %s seconds ---" % (time.time() - start) )
    return
	
	
def splitter(image, size):
	#image=ndarray image
	#shape=image.shape tuple
	#size=height and width of the square chunk 
	#return=list of chunks of the mosaic original image
	shape=image.shape
	if(len(shape)!=3):
		print(shape)
	if(shape[0]<size or shape[1]<size):
		#print("image too small")
		#TODO: implement resize to expand image
		x=(size if (shape[1]<size) else shape[1])
		y=(size if (shape[0]<size) else shape[0])
		new_image=zeros([y,x,shape[2]],dtype=np.float32)
		for Y in range(shape[0]): #shapeY instead of y because we need to transfer only the smaller image into the new bigger image. 
			for X in range(shape[1]): #shapeX instead of x because we need to transfer only the smaller image into the new bigger image. 
				new_image[int(Y+(y-shape[0])/2) , int(X+(x-shape[1])/2)]=image[Y][X]
		image=new_image
		shape=[y,x,shape[2]]
	
	#get number of chunks for each axis
	Y_quantity=int(shape[0]/size + (1 if (shape[0]%size) else 0))
	X_quantity=int(shape[1]/size + (1 if (shape[1]%size) else 0))
	
	#list where the chunks will be saved
	chunks=empty([Y_quantity*X_quantity,size, size, shape[2]],dtype=np.float32)
	
	#iterate
	for Y in range(Y_quantity):
		#print(str(Y/Y_quantity*100)+"%")
		for X in range(X_quantity):
			if(X<X_quantity-1 and Y<Y_quantity-1):
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
				
			chunks[Y*X_quantity+X]=image[tupleY[0]:tupleY[1],tupleX[0]:tupleX[1]]
	#chunks=array(chunks)
	return chunks
		#normal split: split from base+offset to base+offset+size_chunk
		#special split: check if last chunk is full conteined, else the chunk will be sized [Ysize-chunk_size : Ysize],[Xsize-chunk_size : Xsize]
	
def joiner(image, shape, size):
    #IMPORTANT:this code works only when all the chunks of the original image are present
    #it is supposed to work to reassemble the image during a Test when the batch and Epoch sizes don't matter
    
    #the images array after the Noiser process that create the Epoch is cutted and resized in order to fit the requirements. 
    
    #image=ndarray image
    #shape=original shape of the image
    #size=height and width of the square chunk 
    #return=list of chunks of the mosaic original image
    
    sizex=(size if (shape[1]<size) else shape[1])
    sizey=(size if (shape[0]<size) else shape[0])
    
    chunks=(int(shape[0]/size + (1 if (shape[0]%size) else 0)), int(shape[1]/size + (1 if (shape[1]%size) else 0)))
    new_image=zeros([chunks[0]*size,chunks[1]*size, shape[2]],dtype=np.float32)
    
    
    for K in range(chunks[0]):
        for J in range(chunks[1]):
            tmp=image[K*chunks[1]+J]
            if((shape[0]%size or shape[1]%size) or (chunks[0]!=1 or chunks[1]!=1) ):
                
                if (K==chunks[0]-1 and shape[0]%size):
                    tmp=np.roll(tmp,(sizey%size), axis=0)
                   
                   
                if (J==chunks[1]-1 and shape[1]%size):
                    tmp=np.roll(tmp,(sizex%size), axis=1)
                   
            new_image[K*size:(K+1)*size,J*size:(J+1)*size]=tmp
    new_image=new_image[0:sizey , 0:sizex]
    
    #now calculate the min edges to cut:
    if(shape[0]<size or shape[1]<size):
        startY=int((sizey-shape[0])/2) if (shape[0]<size) else 0
        startX=int((sizex-shape[1])/2) if (shape[1]<size) else 0
        new_image=new_image[startY:shape[0]+startY , startX:shape[1]+startX]
    
    return new_image
	

	
def noiser(map, n, batch, miniEpoch, size, num, d, test=False):
	try:
		clean=np.empty([0,size,size,3],dtype=np.float32)
		c=[]
		noise=np.empty([0,size,size,3],dtype=np.float32)
		n=[]
		shape_arr=[]
		for file in map:
			image=asarray(file)
			map[map.index(file)]=0
			image=skimage.util.img_as_float32(image)
			
			shape=image.shape
			shape_arr.append(shape)
			#adding the chunked clean image
			if not test:
				cl_split=splitter(image, size)
			else:
				c.append(splitter(image, size))
			
			
			
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
			
			if not test:
				ns_split=splitter(image, size)
				cl_split, ns_split = shuffle(cl_split, ns_split)
				clean=np.concatenate((clean, cl_split[0:int(cl_split.shape[0]/13)]), axis=0)#,dtype=np.float32
				noise=np.concatenate((noise, ns_split[0:int(ns_split.shape[0]/13)]), axis=0)#,dtype=np.float32
			else:
				n.append(splitter(image, size))
			
			image=0
		#indices = np.indices((3, 3, 3))
		#print(indices)
		#print("hehehe*********************hehehe")
		#np.random.shuffle(indices)
		#print(indices)
		
			
		
		
		#print(str(clean.shape) + "    " + str(noise.shape))
		if not test:
			#clean=shuffle(clean)
			clean, noise = shuffle(clean, noise)
			#if(clean.shape[0]>batch*miniEpoch):
			clean=clean[0:batch*miniEpoch]  #cut the excess out
			noise=noise[0:batch*miniEpoch]
			#print(str(clean.shape) + "    " + str(noise.shape))
			clean=clean.reshape(miniEpoch,batch,size,size,3)
			noise=noise.reshape(miniEpoch,batch,size,size,3)
			d.put([shape_arr,clean,noise])
		else:
			d.put([shape_arr,c,n])
		#print(str(n)+" - iExit")
		return
	
	except Exception:
		print("something went bad on thread " + str(n))
		traceback.print_exc()
	#this is only to implement the progress bar on the threads
	#definitelly not the best solution!
	return


def Writer(input, size, dir_save="/Noise", silence=True):# no longer utilized and need a revisison with the last updates to the code
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
	

	
def Dataset_Builder(batch, epoch, miniEpoch, size=256, path="/Clear", silence=True, test=False):
    # number 	 =  n. of images to process 		100 default
    # size   	 =  size of image in pixel (N x N)		256 default
    # path 		 =  directory where the clear images are located		/Clear folder default
    # silence    =  silence the output. True print only the total time required. False print progress bar, for testing purpose		True default
    # save		 =  save the noisy images on the local directory		False default
    # save_path  =  IF save=True the images will be saved in the specified directory 		/Noise folder default
    # images_per_thread =  number of images per Thread, for load balancing and tuning purpose		20 default
    
    #print(os.cpu_count())
    
    #print("start building batch dataset")
    #indices = np.indices((1, 2, 3))
    #print(indices.shape)
    #print("hehehe*********************hehehe")
    #np.random.shuffle(indices[1:2])
    #print(indices)
    #print(indices.shape)
    #
    #return
    
    
    q=queue.Queue(epoch)
    d=queue.Queue(epoch)
    clean=empty([epoch,miniEpoch,batch,size,size,3],dtype=np.float32)
    noise=empty([epoch,miniEpoch,batch,size,size,3],dtype=np.float32)
    
    start_tot = time.time()
    random.seed(time.time())
    
    executor = ThreadPoolExecutor(max_workers=2)
    if not test:
        executor.submit(loader, q, batch, batch, epoch, miniEpoch, path, size, silence)
    else:
        executor.submit(loader_test, q, batch, path, size, silence)
    
    cores = ThreadPoolExecutor(max_workers=os.cpu_count())
    start = time.time()
    for num in range(epoch):
        data=q.get(True)
        cores.submit(noiser, data, num, batch, miniEpoch, size, num, d, test)
        #print(cores.submit(noiser, data, num, batch, miniEpoch, size, num, d).result())
    shape=0
    if not test:
        for i in tqdm(range(epoch), desc="Workers progress", position=1, disable=silence):
            tmp=d.get(True)
            clean[i]=tmp[1]
            noise[i]=tmp[2]
    else:
        tmp=d.get(True)
        shape, clean, noise = tmp[0], tmp[1], tmp[2]
    if not silence:
        tqdm.write("--- elaboration completed in: %s seconds total---" % (time.time() - start))
    
    cores.shutdown()
    
    executor.shutdown()
    if silence:
        print("--- dataset generation completed in: %s seconds total---" % (time.time() - start_tot))
        
    return shape, clean, noise


