def extractResults(processed):
    results = []
    n = processed.shape[1]

    for i in range(n):
        x = processed[0][i]['facial_area']['x']
        y = processed[0][i]['facial_area']['y']
        w = processed[0][i]['facial_area']['w']
        h = processed[0][i]['facial_area']['h']
        results.append([x,y,w,h])
    return results

def extractFaces(processed):
  faces = []
  n = processed.shape[1]
  for i in range(n):
    faces.append(processed[0][i]['face'])
  return faces

def getFaces(results,image):
  cropped = []
  w = image.shape[1]
  h = image.shape[0]
  for result in results:
      x1 = result[0]
      y1 = result[1]
      x2 = result[0] + result[2]
      y2 = result[1] + result[3]
      cropped.append(image[y1:y2,x1:x2,:])
  return cropped