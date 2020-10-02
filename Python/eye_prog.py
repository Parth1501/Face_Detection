import cv2
import numpy as np

cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml')
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
vid = cv2.VideoCapture(0)
while(True): 
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read()  # ret = 1 if the video is captured; frame is the image
    # Our operations on the frame come here    
    img = cv2.flip(frame,1) 
    
    copy = img.copy()
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    eyes = cascade_eye.detectMultiScale(gray,1.3,5)
    smiles = cascade_smile.detectMultiScale(gray,7.5,5)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(copy,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.putText(copy, 'eyes', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,12), 2)
    for (ex,ey,ew,eh) in smiles:
        cv2.rectangle(copy,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.putText(copy, 'smiles', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,12), 2)
    cv2.imshow('OUTPUT',copy)
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
