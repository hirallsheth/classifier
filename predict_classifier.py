import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import load_model
import cv2

model=load_model("/home/hiral/Documents/model/mymod.h5")
for i in range(1,5):
	train = pd.read_csv('multi_label_train.csv')    # reading the csv file
	train.head() # printing first five rows of the file
	train.columns
	img_name='/home/hiral/Pictures/test_classifier/cropped'+str(i)+'.png'
	actual_image=cv2.imread(img_name,1)
	img = image.load_img(img_name,target_size=(224,224,3))
	img = image.img_to_array(img)
	img = img/255
	classes = np.array(train.columns[2:])
	proba = model.predict(img.reshape(1,224,224,3))
	top_3 = np.argsort(proba[0])[:-4:-1]
	#for i in range(3):
	    #print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
	if((proba[0][top_3[0]])>0.50):
	    print ("Predicted profile:",classes[top_3[0]],(proba[0][top_3[0]]*100))
	plt.imshow(img)
	cv2.imshow('image',actual_image)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()
