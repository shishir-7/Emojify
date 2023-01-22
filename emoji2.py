import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model,Model
from matplotlib import pyplot as plt
import threading
from tensorflow.keras.layers import Input,Activation,Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint



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

emotion_model.load_weights('model_1.h5')


#added content
# input = Input(shape = (100,100,1))

# conv1 = Conv2D(32,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(input)
# conv1 = Dropout(0.1)(conv1)
# conv1 = Activation('relu')(conv1)
# pool1 = MaxPooling2D(pool_size = (2,2)) (conv1)

# conv2 = Conv2D(64,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool1)
# conv2 = Dropout(0.1)(conv2)
# conv2 = Activation('relu')(conv2)
# pool2 = MaxPooling2D(pool_size = (2,2)) (conv2)

# conv3 = Conv2D(128,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool2)
# conv3 = Dropout(0.1)(conv3)
# conv3 = Activation('relu')(conv3)
# pool3 = MaxPooling2D(pool_size = (2,2)) (conv3)

# conv4 = Conv2D(128,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool3)
# conv4 = Dropout(0.1)(conv4)
# conv4 = Activation('relu')(conv4)
# pool4 = MaxPooling2D(pool_size = (2,2)) (conv4)

# flatten = Flatten()(pool4)

# dense_1 = Dense(128,activation='relu')(flatten)

# drop_1 = Dropout(0.2)(dense_1)

# output = Dense(2,activation="sigmoid")(drop_1)
# model = Model(inputs=input,outputs=output)
# model.load_weights('gender_model.h5')
model=load_model('./gender_model.h5')
#model.summary()


cv2.ocl.setUseOpenCL(False)

emotion_dict={0:"  Angry  ",1:"  Disgusted  ",2:"  Fearful  ",3:"  Happy  ",4:"  Neutral  ",5:"  Sad  ",6:"  Surprised  "}
gender_range={0:'male',1:'feamle'}
cur_path=os.path.dirname(os.path.abspath(__file__))
emoji_dist={0:cur_path+"/data/emojis/boy/angry.png",1:cur_path+"/data/emojis/boy/disgusted.png",2:cur_path+"/data/emojis/boy/fearful.png",3:cur_path+"/data/emojis/boy/Happy.png",
4:cur_path+"/data/emojis/boy/neutral.png",5:cur_path+"/data/emojis/boy/sad.png",6:cur_path+"/data/emojis/boy/surprised.png"}

emoji_dist_g={0:cur_path+"/data/emojis/girls/angry.png",1:cur_path+"/data/emojis/girls/disgusted.png",2:cur_path+"/data/emojis/girls/fear.png",3:cur_path+"/data/emojis/girls/happy.png",
4:cur_path+"/data/emojis/girls/neutral.png",5:cur_path+"/data/emojis/girls/sad.png",6:cur_path+"/data/emojis/girls/surprised.png"}

global last_frame1  
output_gender =[0]                                
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]
global frame_number
#models_face=load_model('./model_1.h5')
def show_vid():      
    cap1 = cv2.VideoCapture(0)                                 
    if not cap1.isOpened():                             
        print("cant open the camera1")
    
    # global frame_number
    # length=int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_number+=1
    # if frame_number >= length:
    #     exit()
    # cap1.set(1,frame_number)

    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1,(600,500))
    cv2.imshow('preview',frame1)
    #plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

    bounding_box = cv2.CascadeClassifier('C:/Users/harshita/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]

        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        gender_img=cv2.resize(roi_gray_frame,(100,100), interpolation = cv2.INTER_AREA)
        gender_image_array = np.array(gender_img)
        gender_input = np.expand_dims(gender_image_array, axis=0)
        output_gender[0] = int(np.argmax(model.predict(gender_input)))
        #print(output_gender[0])

        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        show_text[0]=maxindex
    if flag1 is None:
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(1000, show_vid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


def show_vid2():
    
    if gender_range[output_gender[0]] == 'male':
        frame2=cv2.imread(emoji_dist[show_text[0]])
        #print('hii')
    else:
        frame2=cv2.imread(emoji_dist_g[show_text[0]])
        #print('hello')
    pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(frame2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))
    
    lmain2.configure(image=imgtk2)
    root.update()
    lmain2.after(1000, show_vid2)

if __name__ == '__main__':
    root=tk.Tk()   
    # img = ImageTk.PhotoImage(Image.open("logo.png"))
    # heading = Label(root,image=img,bg='black')
    
    # heading.pack() 
    frame_number=0
    heading2=Label(root,text="Photo to Emoji",pady=20, font=('arial',45,'bold'),bg='black',fg='#CDCDCD')                                 
    
    heading2.pack()
    lmain = tk.Label(master=root,padx=50,bd=10)
    lmain2 = tk.Label(master=root,bd=10)

    lmain3=tk.Label(master=root,bd=10,fg="#CDCDCD",bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50,y=250)
    lmain3.pack()
    lmain3.place(x=960,y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900,y=350)
    


    root.title("Photo To Emoji")            
    root.geometry("1400x900+100+10") 
    root['bg']='black'
    exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)
    threading.Thread(target=show_vid).start()
    threading.Thread(target=show_vid2).start()
    root.mainloop()