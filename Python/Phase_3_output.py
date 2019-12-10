import os
import cv2
import pickle
import numpy as np
import pyttsx3
from PIL import Image
engine = pyttsx3.init()
face_classifier=cv2.CascadeClassifier('C:/Users/Zarna/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("trainner.yml")
labels={"person_name":1}
with open("lables.picle","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

speak=1
cap=cv2.VideoCapture(0)
prev=[]
n=1.8
while True:
    ret,frame=cap.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,n,3)
    for (x, y, w, h) in faces:

        roi_gray = gray[y:y + h, x:x + w]
        roi_color=frame[y:y + h, x:x + w]
        if speak == 1:
            cv2.imwrite("1.jpg", roi_color)
            im = Image.open("1.jpg", "r")
            pix_val = list(im.getdata())
            pix_val_flat = [x for sets in pix_val for x in sets]
           # n=((pix_val_flat[0]*0.3) + (pix_val_flat[1]*0.59) + (pix_val_flat[2]*0.11))
            speak = 0
        print(pix_val_flat)
        color=(0,0,255)
        stroke=2

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
        id_,conf=rec.predict(roi_gray)
        conf=(1-(conf/300))*100
        print(conf)
        if conf<80 and n>=1.1:
            n-=0.1
        elif n<=1.8 and conf<80:
            n+=0.1
        else:
            if conf>=82 :
                #print(id_)
                #print(labels[id_])
                print(n)
                font=cv2.FONT_HERSHEY_COMPLEX
                name=labels[id_]
                color = (25, 200, 215)
                stroke=2
                cv2.putText(frame,name+ str(round(conf)),(x-10,y-10),font,1,color,stroke)
                if name not in prev:
                    engine.say("Hello"+ name)
                    engine.runAndWait()
    #                speak=0
            else:
                font = cv2.FONT_HERSHEY_COMPLEX
                name = "Unknown"
                color = (25, 200, 215)
                stroke = 2
                cv2.putText(frame, name + str(round(conf)) + "%", (x - 10, y - 10), font, 1, color, stroke)
            if name  not in prev:
                prev.append(name)
    cv2.imshow('Face',frame)
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()