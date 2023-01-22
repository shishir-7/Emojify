from statistics import mode
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.layers import Input,Activation,Add,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model,Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
path="./data/test/"
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))



# no_of_classes = 7
# model = Sequential()

# #1st CNN layer
# model.add(Conv2D(64,(3,3),padding = 'same',input_shape = (48,48,1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.25))

# #2nd CNN layer
# model.add(Conv2D(128,(5,5),padding = 'same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout (0.25))

# #3rd CNN layer
# model.add(Conv2D(512,(3,3),padding = 'same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout (0.25))

# #4th CNN layer
# model.add(Conv2D(512,(3,3), padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())

# #Fully connected 1st layer
# model.add(Dense(256))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.25))

# model.add(Dense(512))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.25))

# model.add(Dense(no_of_classes, activation='softmax'))

emotion_model.load_weights('model_1.h5')
test_data=[]
y_test=[]
y_pred=[]
pred=[]
emotion_dict={0:"angry",1:"disgust",2:"fear",3:"happy",4:"neutral",5:"sad",6:"surprise"}
for i in range(0,7):
    path_1=path+emotion_dict[i]
    for img in os.listdir(path_1):
        img = cv2.imread(str(path_1)+"/"+str(img))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(48,48))
        img=np.expand_dims(np.expand_dims(img, -1), 0)
        test_data.append(np.array(img))
        y_test.append(i)
n=len(test_data)
for i in range(0,len(test_data)):
    output=emotion_model.predict(test_data[i])
    output=int(np.argmax(output))
    y_pred.append(output)
    if output==y_test[i]:
        pred.append(1)
    else:
        pred.append(0)
print("Accuracy")
acc=sum(pred)/n
print(acc)
cm=confusion_matrix(y_test,y_pred)      
print(cm)
