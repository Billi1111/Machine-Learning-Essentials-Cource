# train the classifier to learn who is the person (classification)
import cv2
import numpy as np 
import os 
#dataset path "make folder name it dataset and put the path the folder must contain at least 2 files from the previous code"
dataset_path=r"D:\Bilal\Learning\Udemy\Machine learning essentials\section 9\dataset"
faceData=[]
labels= []
nameMap={}

classId = 0

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        #removing last 4 letters Bilal.npy to Bilal
        nameMap[classId]= f[:-4]
        # x value
        dataItem=np.load(os.path.join(dataset_path , f))
        m=dataItem.shape[0]
        faceData.append(dataItem)
        
        #y value
        target= classId * np.ones((m,))
        classId+=1
        labels.append(target)


#to let the program make data in one column ( , ) to ( )
Xt=np.concatenate(faceData, axis=0)
yt=np.concatenate(labels, axis=0).reshape((-1,1))

print(Xt.shape)
print(yt.shape)
print(nameMap)

#algorithm
def dist(p,q):
    return np.sqrt(np.sum((p-q)**2))
def knn(X,y,xt,k=5):
    m=X.shape[0]
    dlist=[]
    
    for i in range(m):
        d = dist(X[i],xt)
        dlist.append((d,y[i]))
    
    dlist= sorted(dlist)
    dlist = (dlist[:k])
    labels = np.array([label for dist, label in dlist])

    labels, cnts = np.unique(labels, return_counts = True)
    idx = cnts.argmax()
    pred = labels[idx]

    return int(pred)

#predictions
#create camera object
cam=cv2.VideoCapture(0)
#model
model= cv2.CascadeClassifier(r"D:\Bilal\Learning\Udemy\Machine learning essentials\section 9\haarcascade_frontalface_alt.xml")
#add offset for face cropping
offset = 10

while True:
    sucess, img = cam.read()
    if not sucess:
        print("Error: Could not read image")
        continue
    
    faces=model.detectMultiScale(img, 1.3,5)
    
    #render a box around each face and predict its name
    for f in faces:
        x,y,w,h=f  # use f instead of faces[0]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        #crop and save the largest face
        cropped_face = img[y-offset : y + h + offset, x - offset : x + offset + w]
        cropped_face=cv2.resize(cropped_face,(100,100))
       
        #predict the name using knn
        classPredicted=knn(Xt,yt,cropped_face.flatten())# we flatten because above the shape is (100,100,3) and we want to flatten the image
        namePredicted= nameMap[classPredicted]
        print(namePredicted)
        # display name and box
        cv2.putText(img, namePredicted, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,0),2,cv2.LINE_AA)
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)

    cv2.imshow("Prediction Window", img)
 
    key= cv2.waitKey(1)#time in milliseconds (0)
    if key==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
