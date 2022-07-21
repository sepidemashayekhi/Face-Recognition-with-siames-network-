from matplotlib import image
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from l1Dist import  L1Dist
import cv2 
import numpy as np 
import os 
import tensorflow as tf 


model =load_model('my_model.h5', 
                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
print('load model ....')

def process(image_path):
    image=tf.io.read_file(image_path)
    image=tf.io.decode_jpeg(image)
    image=tf.image.resize(image,(100,100))
    image=np.expand_dims(image,axis=0)
    return image


def Verification(image1_path,image2_path):
    global model
    image1=process(image1_path)
    image2=process(image2_path)
    result=model.predict([image1, image2])
    return result
anchor_file='C:/Users/Mashayekhi/Desktop/itsaaz task/anc'
positive_file='C:/Users/Mashayekhi/Desktop/itsaaz task/pos'


image_path=os.path.join(anchor_file,os.listdir(anchor_file)[350])
positive_path=os.path.join(positive_file,os.listdir(positive_file)[350])

result=Verification(image_path,positive_path)
print(result)