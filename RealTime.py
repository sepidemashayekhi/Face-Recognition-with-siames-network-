import cv2
import os
import numpy as np 
from l1Dist import  L1Dist
from tensorflow.keras.models import load_model
import tensorflow as tf
from random import randint

databaseDir='DataBase'
Threshold=80

cascade_model=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model =load_model(filepath='my_model.h5',
                            custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
print('Load model ..........')

database={
    0:'Aron',
    1:'Adriana',
    2:'Aishwar',
    3:'Alessa',
    4:'Ali',
    5:'sepide'
}

def preprocess_frames(image):
    image=tf.image.resize(image,(100,100))
    image=tf.expand_dims(image,axis=0)
    image=image/255
    return image
def load_image_database(image_path):
    image=tf.io.read_file(image_path)
    image=tf.io.decode_jpeg(image)
    image=tf.image.resize(image,(100,100))
    image=tf.expand_dims(image,axis=0)
    return image
def Verification(face_frame,image_path):
    global model
    face_frame=preprocess_frames(face_frame)
    database_img=load_image_database(image_path)
    result=model.predict([face_frame,database_img])[0]
    return result


cam=cv2.VideoCapture(0)

while True:
    _,frame=cam.read()
    frame=cv2.flip(frame,1)
    face_loc=cascade_model.detectMultiScale(frame)
    for bbox in face_loc:
        Color=(randint(0,254),randint(0,255),randint(0,255))
        cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),Color,2,1)
    
    cv2.imshow('frame',frame)
    k=cv2.waitKey(1)
   
    if k ==27:
        break
   
    elif k==ord('a'):
        
        if len(face_loc)!=0:
            results=[]
            face=frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
            cv2.imshow("face",face)
           
            for name_image in os.listdir(databaseDir):
                image_path=os.path.join(databaseDir,name_image)
                result=Verification(face,image_path)
                result=list(result)
                results.append(result)
            number=np.argmax(results)
            
            if int(results[number] )>1:
                print("3333")
           
            name_employee=database[number]
            print('{}'.format(name_employee),' is peresent')
            
            # cv2.destroyWindow("face")
        elif len(face_loc)==0:
            print('There is a no face')    
