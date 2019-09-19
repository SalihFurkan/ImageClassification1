import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time 


def preprocessImage(image):
    

	#convert opencv numpy array to PIL image

	#PADDING
	    
	#get size of image
	x,y = image.size

	#get maximum one to extend other edge to same length to make image square
	square_edge = max(x,y)

	#create just black image with size of square_edge X square_edge
	padded_image = Image.new('RGB', (square_edge, square_edge), (0,0,0))


	#copy available image into black square image to get padded version of image
	#In Python 3, to get an integer result from a division you need to use // instead of /
	up_left_coordinate = ((square_edge - x) // 2, (square_edge - y) // 2)


	#Note: in this library, cartesian pixel coordinate system with (0,0) in the upper left corner is used
	# so, to centre old image on square padded image, corresponding upperleft coordinate would be obtained by the above formula

	padded_image.paste(image, up_left_coordinate)

	image = padded_image




	#RESCALING TO 224 X 224

	RESCALE_SIZE = 224


	#utilize resize function of pillow library
	rescaled_image = image.resize((RESCALE_SIZE,RESCALE_SIZE))
	image = rescaled_image



	#CONVERT IMAGES INTO NUMPY ARRAY BEFORE NORMALIZATION


	image = np.asarray(image).astype(np.float32)




	#DO NORMALIZATION
	image = np.true_divide(image,255)

	image[:,:,0] = np.subtract(image[:,:,0],0.485)
	image[:,:,1] = np.subtract(image[:,:,1],0.456)
	image[:,:,2] = np.subtract(image[:,:,2],0.406)

	image[:,:,0] = np.true_divide(image[:,:,0],0.229)
	image[:,:,1] = np.true_divide(image[:,:,1],0.224)
	image[:,:,2] = np.true_divide(image[:,:,2],0.225)

	processed_image = image



	return processed_image

#function ends ------------------------



start = time.time()



#READING
training_images = []
labels =[]
labelNum = 0
label = 0
#trace all subfolders in train folder
for folder_name in sorted(glob.glob("train/*")):
	label = label +1
	labelNum = labelNum + 1

	#get all images placed in each folder
	for image_path in sorted(glob.glob( folder_name + "/*.JPEG")):
		
		#read image
		image = Image.open(image_path).convert("RGB") # load an image_path
		
		#add image to list
		training_images.append(image)

		labels.append(label)






#PREPROCESSING

#Preprocess each image

normalized_array = []

for image in training_images:
	
	normalized_array.append(preprocessImage(image))






#FEATURE EXTRACTION

import torch
from resnet import resnet50
from sklearn import preprocessing

feature_vectors = []


model = resnet50(pretrained=True)
model.eval()

for image in normalized_array:

	# we append an augmented dimension to indicate batch_size, which is one
	image = np.reshape(image, [1, 224, 224, 3])


	# model takes as input images of size [batch_size, 3, im_height, im_width]
	image = np.transpose(image, [0, 3, 1, 2])
	#(For the second arguement) i in the j-th place in the tuple means a’s i-th axis becomes a.transpose()’s j-th axis.
	#e.g. shape of the new array would be [1,3,224,224]


	# convert the Numpy image to torch.FloatTensor
	tensor_image = torch.from_numpy(image)


	# extract features
	feature_vector = model(tensor_image)


	# convert the features of type torch.FloatTensor to a Numpy array
	# so that you can either work with them within the sklearn environment
	# or save them as .mat files

	
	feature_vector.data.cpu().numpy()

	feature_vector = feature_vector.data.numpy()
	#Note: the version placed in the project document ends up with an error
	

	#before adding list, normalize feature vector
	feature_vector = preprocessing.normalize(feature_vector,norm='l2')

	feature_vector = np.squeeze(feature_vector)


	feature_vectors.append(feature_vector)

	



#TRAINING

#for each class, we create a binary svm classifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

import pickle

svm_classifiers = []

for i in range(1,labelNum+1):
	
	
	#create binary labels for svm classifier
	binarylabels = np.asarray(labels)

	binarylabels[binarylabels!=i] = 0
	
	# parameters to be optimized through cross validation while fitting data
	parameters = {'kernel':('linear', 'rbf','poly'), 'C':[0.1,1, 10,100]}

	estimator = SVC(gamma='auto',probability=True,random_state=0)

	svm_model = GridSearchCV(estimator, parameters, cv=5)

	svm_model.fit(feature_vectors,binarylabels)

	svm_classifiers.append(svm_model)







	

end = time.time()


i = 0

for model in svm_classifiers:
	
	i = i+1	
	
	filename = 'model' + str(i) + '.sav'
	pickle.dump(model, open(filename, 'wb'))

print("total execution time : " + str(end - start))