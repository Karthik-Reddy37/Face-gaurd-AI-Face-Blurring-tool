import copy 
import cv2

def crop(result,image):
  x1 = result[0]
  y1 = result[1]
  x2 = result[0]+result[2]
  y2 = result[1]+result[3]
  cropped = image[y1:y2,x1:x2,:]
  return cropped

def blur_section(cropped):
  blurred = []
  for section in cropped:
    blurred.append(cv2.blur(section,(20,20)))
  return blurred

def back(result,image,blurred):
  w = image.shape[1]
  h = image.shape[0]
  x1 = result[0]
  y1 = result[1]
  x2 = result[0]+result[2]
  y2 = result[1]+result[3]
  image[y1:y2,x1:x2,:] = blurred
  return image

def generateBlurred(image,results,blurFaces):
  duplicate = copy.copy(image)
  final_image = copy.copy(image)
  for result in results:
    if result[0] in blurFaces:
      cropped = crop(result[1],final_image)
      blurred = blur_section(cropped)
      final = back(result[1],final_image,blurred)

  return final_image