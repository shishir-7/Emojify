from statistics import mode
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model,Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
path = "./Gender_data/UTKFace"
pixels = []
#age = []
gender = [] 

i=0
for img in os.listdir(path):
  i=i+1
  genders = img.split("_")[1]
  img = cv2.imread(str(path)+"/"+str(img))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  img=cv2.resize(img,(100,100))
  pixels.append(np.array(img))
  gender.append(np.array(genders))

pixels = np.array(pixels)
gender = np.array(gender,np.uint64)

x_train,x_test,y_train,y_test = train_test_split(pixels,gender,random_state=100)
model=load_model('./gender_model.h5')
n=len(x_test)
pred=[]
predicted_value=[]
for i in range(0,n):
  output=int(np.argmax(model.predict(np.expand_dims(x_test[i],axis=0))))
  predicted_value.append(output)
  if output==y_test[i]:
    pred.append(1)
  else:
    pred.append(0)

cm=confusion_matrix(y_test,predicted_value)
print("Accuracy")
acc=sum(pred)/n
print(acc)
print(cm)