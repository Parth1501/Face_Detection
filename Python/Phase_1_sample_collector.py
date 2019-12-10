import cv2
import numpy as np
face_classifier=cv2.CascadeClassifier('C:/Users/Zarna/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
def face_rec(img,n) :
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=face_classifier.detectMultiScale(gray,n,3)
    if face is ():
        return None
    for(x,y,w,h) in face:
        crop=img[y:y+h,x:x+w]
    return crop

capture=cv2.VideoCapture(0)
count=800

while True:
    n=1.1
    ret, frame=capture.read()
    if face_rec(frame,n) is not None:
        count+=1;

        face=cv2.resize(face_rec(frame,n),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_path="E:\Parth\sample\img  "+str(count)+".jpg"
        cv2.imwrite(file_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
        if count>10:
            n=1.2
        elif count>20:
            n=1.3
        elif count>30:
            n=1.4
        elif count>40:
            n=1.5
        else:
            n=1.6
    else:
        print('Face not recognizing')
        pass
    if cv2.waitKey(1)==13:
        break

capture.release()
cv2.destroyAllWindows()
print("Succesful!")

