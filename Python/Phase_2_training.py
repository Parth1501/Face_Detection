import os
import cv2
import pickle
import numpy as np
from PIL import Image
face_classifier=cv2.CascadeClassifier('C:/Users/Zarna/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
rec=cv2.face.LBPHFaceRecognizer_create()
Base_dir= os.path.dirname("E:\Parth\ ")
img_dir=os.path.join(Base_dir,"Sample")
current_id=0
label_ids={}
y_labels=[]
x_train=[]
count=0
n=1.1
for root,dirs,files in os.walk(img_dir):
    count=0
    for file in files:
        if file.endswith("jpg"):
            path=os.path.join(root,file)
            label=os.path.basename(root).replace(" ","-")
            print(label)
            if label in label_ids:
                pass
            else:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            print(label_ids)
           #y_labels.append(label)
            #x_train.append(path)
            pil_image=Image.open(path)
            image_array=np.array(pil_image,"uint8")
            #print(image_array)
            faces=face_classifier.detectMultiScale(image_array,n,3)
            count+=1
            if count > 10:
                n = 1.2
            elif count > 20:
                n = 1.3
            elif count > 30:
                n = 1.4
            elif count > 40:
                n = 1.5
            else:
                n = 1.6
            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)

with open("lables.picle","wb") as f:
    pickle.dump(label_ids,f)

rec.train(x_train,np.array(y_labels))
rec.save("trainner.yml")
