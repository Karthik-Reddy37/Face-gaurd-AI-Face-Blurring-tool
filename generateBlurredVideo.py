import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
import keras.optimizers as ko
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow
import torch
import cv2
import os
import copy

from deepface import DeepFace
from deepface.commons import functions

from compare import getUnique
from blurrTools import generateBlurred


## defining the model name to build the model
model_name = "VGG-Face" #"Facenet" 
target_size = functions.find_target_size(model_name = model_name)
model = DeepFace.build_model(model_name = model_name)

video_file = "ppl_running.mp4"   # the video file which has to be blurred
filename = "new.avi"     # the file name to be created once blurred
uniqueFaces = []

# starting cap and video writer
cap  = cv2.VideoCapture(video_file)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fcount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)


video_writer = cv2.VideoWriter(os.path.join(filename),cv2.VideoWriter_fourcc('P','I','M','1'),fps,(width,height))

detected = []
framegap = 10
i = 0

while cap.isOpened():
    ret, frame = cap.read()
         # if frame is read correctly ret is True
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break
    
    if i%framegap == 0:
      newData = getUnique(uniqueFaces,frame,model,target_size)
      detected.append(newData)
      # try:
      #   newData = getUnique(uniqueFaces,frame,model,target_size)
      #   detected.append(newData)
      # except:
      #   detected.append([[-1]])
      #   print("face not detected")

    #frames.append(frame)
    i+=1
    print(i)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()

rows = 3
columns = 3
fig = plt.figure(figsize=(10, 6))
start = 0
for i in range(int(len(uniqueFaces))):
  img = uniqueFaces[i]
  fig.add_subplot(rows, columns, i-start+1)
  plt.imshow(img)
  plt.axis('off')
  plt.title("face no. " + str(i))

plt.savefig("results.jpg")

blurFaces = set()

while True:
  inp = input("enter no. of the faces to blur else enter No :")

  if inp == "No":
    break
  else:
    blurFaces.add(int(inp))
  


def writeVideo(frame,detected,blurFaces,index):
  if index<len(detected) and detected[index][0][0] != -1:
    img = generateBlurred(frame,detected[index],blurFaces)
  else:
    img = frame
  
  
  video_writer.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

framecount = 1
index = 0
cap  = cv2.VideoCapture(video_file)

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fcount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)

video_writer = cv2.VideoWriter(os.path.join(filename),cv2.VideoWriter_fourcc('P','I','M','1'),fps,(width,height))

while cap.isOpened():
    ret, frame = cap.read()
         # if frame is read correctly ret is True
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break

    writeVideo(frame,detected,blurFaces,index)
    
    if framecount%framegap == 0:
      index+=1

    framecount+=1
    print(framecount)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

video_writer.release()
cap.release()
cv2.destroyAllWindows()