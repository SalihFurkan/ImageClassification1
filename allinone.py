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







#TESTING


import glob

import cv2 as cv
import numpy as np
import sys

import torch
from resnet import resnet50
from sklearn import preprocessing
from PIL import Image


model = resnet50(pretrained=True)
model.eval()

#READ TEST IMAGES
test_images = []


#read all images under the test/images folder
i = range(0,100)
for image_path in i:
    
    #read image
    image = cv.imread("test/images/"+str(image_path)+".JPEG") # load an image_path
    
    print("test/images/"+str(image_path)+".JPEG")

    #convert image to rgb
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    #add image to list
    test_images.append(image)   



# 1.Extract candidate windows

modelPath = "model.yml"
windows = []
i = 1
#for each image extract window
for image in test_images:

    edge_detection = cv.ximgproc.createStructuredEdgeDetection(modelPath)
    rgb_im = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(50)

    #get boxes values consist of x,y,width,height
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    windows.append(boxes)
    



# 2.Classification and localization
optimum_windows = np.asarray([],dtype=np.int8)
predictions = np.asarray([],dtype=np.int8)


#for each image 
for i in range(0,len(windows)):

    test_image = test_images[i]

    candidate_windows = windows[i]

    


    windows_predictions = np.asarray([],dtype=np.int8)
    windows_probabilities= np.asarray([]) 
    a = []
    #for each window
    for j in range(0,len(candidate_windows)):

        #get coordinates and length values
        a = candidate_windows[j]
        x, y, w, h = candidate_windows[j]

        #crop image to window obtained by box edge method
        #Note: opencv images are stored in numpy array, top left point corresponds to 0,0 coordinate
        cropped_image = test_image[y:y+h,x:x+w,:]
        #convert opencv numpy array to PIL image
        cropped_image = Image.fromarray(cropped_image.astype('uint8'), 'RGB')

        cropped_image = preprocessImage(cropped_image)

        # we append an augmented dimension to indicate batch_size, which is one
        cropped_image = np.reshape(cropped_image, [1, 224, 224, 3])


        # model takes as input images of size [batch_size, 3, im_height, im_width]
        cropped_image = np.transpose(cropped_image, [0, 3, 1, 2])
        #(For the second arguement) i in the j-th place in the tuple means a’s i-th axis becomes a.transpose()’s j-th axis.
        #e.g. shape of the new array would be [1,3,224,224]


        # convert the Numpy image to torch.FloatTensor
        tensor_image = torch.from_numpy(cropped_image)



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


    
        classifier_scores = np.asarray([])
        
        #apply each of binary svm classifier to window
        for classifier in svm_classifiers:
            
            #get window score for each binary classifier            
            classifier_scores = np.append(classifier_scores,classifier.predict_log_proba(feature_vector.reshape(1,-1))[0][1])
            
            #Note: (error)array.reshape(1, -1) if it contains a single sample
    
        window_prediction = np.argmax(classifier_scores)+1  
        window_probability = np.amax(classifier_scores)
        
        windows_predictions = np.append(windows_predictions,window_prediction)
        windows_probabilities = np.append(windows_probabilities,window_probability)

    image_prediction = windows_predictions[np.argmax(windows_probabilities)]
   
    optimum_window = candidate_windows[np.argmax(windows_probabilities)]
    
    print(optimum_window)

    optimum_window = np.reshape(optimum_window,4)

    optimum_windows = np.append(optimum_windows,[optimum_window])
    
    predictions = np.append(predictions,image_prediction) 


label_dictionary = {1:'n01615121', 2:'n02099601', 3:'n02123159',4:'n02129604', 5:'n02317335', 6:'n02391049', 7:'n02410509', 8:'n02422699', 9:'n02481823',10:'n02504458'}


prediction_labels = [label_dictionary[label] for label in predictions]

optimum_windows = optimum_windows.astype(int).reshape(len(test_images),4)

np.savetxt("windows.txt",optimum_windows,delimiter = ',')
with open('predictions.txt', 'w') as f:
    for item in prediction_labels:
        f.write("%s\n" % item)






#EVALUATION

lines = [line.rstrip('\n') for line in open('test/bounding_box.txt')]

test_labels = [line.split(',')[0] for line in lines]
x1_test = [int(line.split(',')[1]) for line in lines]
y1_test = [int(line.split(',')[2]) for line in lines]
x2_test = [int(line.split(',')[3]) for line in lines]
y2_test = [int(line.split(',')[4]) for line in lines]






#6.1 Classification accuracy:

class_test_size = np.zeros(10)

true_class_prediction = np.zeros(10)

false_class_prediction = np.zeros(10)

total_prediction_num = 0

total_true_num = 0


for prediction,label in zip(prediction_labels,test_labels):

	inverse_label_dict = {'n01615121':0, 'n02099601':1, 'n02123159':2, 'n02129604':3, 'n02317335':4, 'n02391049':5, 'n02410509':6, 'n02422699':7, 'n02481823':8, 'n02504458':9}
	
	prediction = inverse_label_dict[prediction] 
	label = inverse_label_dict[label] 

	if prediction == label:

		total_true_num = total_true_num + 1

		true_class_prediction[prediction] = true_class_prediction[prediction] + 1

	else:

		false_class_prediction[prediction] = false_class_prediction[prediction] + 1


	class_test_size[label] = class_test_size[label] + 1

	total_prediction_num = total_prediction_num + 1


label_dictionary = {0:'n01615121', 1:'n02099601', 2:'n02123159',3:'n02129604', 4:'n02317335', 5:'n02391049', 6:'n02410509', 7:'n02422699', 8:'n02481823',9:'n02504458'}

for key in label_dictionary:

	label = label_dictionary[key]

	print("For the class  " + label)

	print("Confusion matrix :")
	print("True Positives: " + str(true_class_prediction[key]))
	print("False Positive: " + str(false_class_prediction[key]))
	print("True Negative: " + str((total_prediction_num - (true_class_prediction[key]+false_class_prediction[key])) - (class_test_size[key]-true_class_prediction[key])))
	print("False Negative: " + str( class_test_size[key]-true_class_prediction[key]))

	print("Precision : " + str(true_class_prediction[key]/(true_class_prediction[key]+false_class_prediction[key])))
	print("Recall : " + str(true_class_prediction[key]/class_test_size[key]))

	print ("-----------------------------------------------")

print("Overall accuracy: " + str(total_true_num/total_prediction_num))





#6.2 Localization accuracy
true_localized_num = 0

for window_predicted,x1t,y1t,x2t,y2t, prediction,label in zip(optimum_windows,x1_test,y1_test,x2_test,y2_test,prediction_labels,test_labels):

	x1p,y1p,width,height = window_predicted
	x2p = x1p + width
	y2p = y1p + height

	predicted_window_area = (y2p-y1p) * (x2p-x1p)
	test_window_area = (y2t-y1t) * (x2t - x1t)
	
	intersection_point1 = [max(x1p,x1t),max(y1p,y1t)]
	intersection_point2 = [min(x2p,x2t),min(y2p,y2t)]

	intersection_area = float(intersection_point2[1]-intersection_point1[1]) * float(intersection_point2[0]-intersection_point1[0])

	portion = (intersection_area / (float(predicted_window_area+test_window_area)-intersection_area))

	if portion>0.50 :

		true_localized_num = true_localized_num + 1



print("Localization accuracy: " + str(true_localized_num/total_prediction_num))



 





end = time.time()

print("total execution time : " + str(end - start))