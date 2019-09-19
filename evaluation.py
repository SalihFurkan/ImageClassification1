import numpy as np

prediction_labels = [line.rstrip('\n') for line in open('predictions.txt')]

window_lines = [line.rstrip('\n') for line in open('windows.txt')] 

optimum_windows = [[int(float(line.split(',')[0])),int(float(line.split(',')[1])),int(float(line.split(',')[2])),int(float(line.split(',')[3]))] for line in window_lines]

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

