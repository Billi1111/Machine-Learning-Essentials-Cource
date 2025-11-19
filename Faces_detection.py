#Read a Video from Web cam using opencv
# Face detection in video 
# Click 20 pictures of the person who comes in the front of the camera and save them as numpy
import cv2
import numpy as np
#create camera object
cam=cv2.VideoCapture(0)
fileName= input("Enter the name of the person: ")
dataset_path= r"D:\Bilal\Learning\Udemy\Machine learning essentials\section 9\dataset"
offset = 20
model= cv2.CascadeClassifier(r"D:\Bilal\Learning\Udemy\Machine learning essentials\section 9\haarcascade_frontalface_alt.xml")
#Create a list to save face data
faceData=[]
skip=0

#read image from camera object
while True:
   sucess, img = cam.read()
   if not sucess:
      print("Error: Could not read image")
      #store the gray images
   grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


   faces=model.detectMultiScale(img, 1.3,5)
   #pick the face with the largest bounding box
   faces=sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
   #pick the largest face
   if len(faces)>0:
     # f = faces[-1:]

   #for (x,y,w,h) in faces[-1:]:
      x,y,w,h=faces[0]
      cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
   #crop and save the largest face
      cropped_face = img[y- offset : y + h + offset, x - offset : x + offset+ w]

      cropped_face=cv2.resize(cropped_face,(100,100))
      skip+=1

      if skip %10 == 0:
         faceData.append(cropped_face)
         print("saved so far",str(len(faceData)))

   cv2.imshow("Image", img)
  # cv2.imshow("Cropped Face", cropped_face)
   if not sucess:
      print("Error: Could not read image")
 
    
   key= cv2.waitKey(1)#time in milliseconds (0)
   if key==ord('q'):
      break
#write the faceData on the disk
faceData= np.asarray(faceData)

m=faceData.shape[0]
faceData= faceData.reshape((m,-1))
print(faceData.shape)
#save on the disk as np array
filepath=dataset_path + "\\" + fileName+ ".npy"
np.save(filepath, faceData)
print("Data Saved Successfully" + filepath)
  
cam.release()
cv2.destroyAllWindows()