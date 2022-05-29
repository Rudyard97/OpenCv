import operator
from turtle import width
import cv2
from cv2 import CascadeClassifier
from cv2 import cvtColor


face_cascade=CascadeClassifier("haarcascade_frontalface_alt2.xml")
profile_cascade=cv2.CascadeClassifier("haarcascade_profileface.xml")
cap=cv2.VideoCapture(0)
width=int(cap.get(3))
marge=60

while True:
    ret, frame=cap.read()
    tab_face=[]
    tickmark=cv2.getTickCount()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    for x,y,w,h in face:
        tab_face.append([x, y, x+w, y+h])
    face=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    for x,y,w,h in face:
        tab_face.append([x, y, x+w, y+h])
    gray2=cv2.flip(gray, 0)
    face=face_cascade.detectMultiScale(gray2, scaleFactor=1.2, minNeighbors=3)
    for x,y,w,h in face:
        tab_face.append([width-x, y, width-(x+w), y+h])
    tab_face=sorted(tab_face, key=operator.itemgetter(0, 1))
    index=0
    for x,y,x2,y2 in tab_face:
        if not index or (x-tab_face[index-1][0]>marge or y-tab_face[index-1][1]>marge):
          cv2.rectangle(frame, (x, y),(x2, y2), (255,0,0),2)
        index+=1
    if cv2.waitKey(1)==ord('q'):
        break
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
    cv2.putText(frame, "FPS: {:05.2f}".format(fps),(10,30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)    
    cv2.imshow('Test1' , frame)
cap.release()
cv2.destroyAllWindows()    
