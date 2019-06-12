
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
print ("hey")
train = pd.read_csv('multi_label_train.csv')    # reading the csv file
train.head() # printing first five rows of the file
train.columns

train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('/home/hiral/Desktop/img_directory/'+train['Id'][i]+'.jpg',target_size=(224,224,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)
X.shape
plt.imshow(X[2])
train['Profile'][2]
y = np.array(train.drop(['Id', 'Profile'],axis=1))
y.shape
print ("hey")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(3, 3), activation="relu", input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)
model.save("/home/hiral/Documents/model/mymod.h5")
