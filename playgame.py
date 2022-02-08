import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import sleep
import pathlib
import random
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

print("import OK")

#TesnoFlow memory
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

def make_pipeline():
    s = ('v4l2src device=/dev/vide0 !  num-buffers=1 ! jpegenc ! filesink location=/home/alberto/sasso/mano.jpg')
    return Gst.parse_launch(s)

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('rps.h5')

# Show the model architecture
new_model.summary()

ultron_hand = ["Paper","Scissors","Rock"]

while True:
     sleep(20)
     #Take photo
     os.system('gst-launch-1.0 v4l2src num-buffers=1 ! jpegenc ! filesink location=/home/alberto/sasso/mano.jpg')

     img = mpimg.imread('mano.jpg')
     imgplot = plt.imshow(img)
     plt.show(block=False)
     plt.pause(3)
     plt.close()
     img.resize((150,150))

     img_array = tf.keras.preprocessing.image.img_to_array(img)
     data_images = np.expand_dims(img_array, axis=0)

     print("Symbolic tensor",img_array)
     data_images = np.vstack([data_images])
     data_images = np.tile(data_images, 3)
     print("data_images",data_images)
     classes = new_model.predict(data_images, batch_size=10)
     print("Classes : ",classes)
     print(round(classes[0][0]),round(classes[0][1]),round(classes[0][2]))

     if round(classes[0][0]) == 1:
        print ("Rock")
        hand = -3
     if round(classes[0][1]) == 1:
        print("Scissors")
        hand = -2
     if round(classes[0][2]) == 1:
        print("Paper")
        hand = -1

     ultron_choice =  random.randrange(1, 3)
     print("Ultron has chooised ",ultron_hand[ultron_choice-1])

     result = hand + ultron_choice

     if result == 0:
        print("Tie !!!")
     if result == 1 or result == -2:
        print ("I won, loser ;-)")
     if result == -1 or result == 2:
        print ("S*#!t, I lost")


