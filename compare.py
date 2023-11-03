import numpy as np
from deepface import DeepFace
from deepface.commons import functions
from results import extractResults,extractFaces,getFaces
from scipy.spatial import distance
import cv2

def euclidDist(img1_representation,img2_representation):
  distance_vector = np.square(np.array(img1_representation) - np.array(img2_representation)) #getting distance from representation
  distance = np.sqrt(distance_vector.sum())
  return distance

def cosDist(img1_representation,img2_representation):
  return distance.cosine(img1_representation,img2_representation)

def compareFaces(image1,image2,model):
  img1_representation = model.predict(image1).tolist()
  img2_representation = model.predict(image2).tolist()
  if cosDist(img1_representation,img2_representation) <=0.31:
    return True
  return False


## Getting the unique faces in the image and store them in a list

def getUnique(uniqueFaces,image,model,target_size):
    facedata = []
    # getting the results as faces from the detector
    initial_results = DeepFace.extract_faces(image, target_size=target_size,detector_backend ="retinaface")  #detector_backend ="mtcnn"
    initial_results = np.expand_dims(initial_results, axis=0)
    
    #extracting the different results
    results = extractResults(initial_results)
    faces = extractFaces(initial_results)
    
    # comparing the faces to the faces already in the list
    i = 0 
    for frameFace in faces:
        frameFace = np.expand_dims(frameFace, axis=0)
        if len(uniqueFaces) == 0:
            face = cv2.cvtColor(frameFace[0],cv2.COLOR_BGR2RGB)
            uniqueFaces.append(face)
            facedata.append([0,results[i]]) #facedata[0].append(results[i]) #
            
        else: 
            check = True
            j = 0
            for existingFace in uniqueFaces:
                existingFace = np.expand_dims(existingFace, axis=0)
                #j+=1
                if check and compareFaces(frameFace,existingFace,model):
                    check = False
                    facedata.append([j,results[i]]) #facedata[j].append(results[i])
                    break
                j=j+1 
                
            if check:
                face = cv2.cvtColor(frameFace[0],cv2.COLOR_BGR2RGB)
                uniqueFaces.append(face)
                facedata.append([j,results[i]]) #facedata[j].append(results[i]) 
        
        i= i+1
        
    return facedata