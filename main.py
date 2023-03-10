import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Image_Attendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cls in myList:
    curImg =  cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

with open('Script.txt','r') as f:
    scr = f.read()

print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = findEncodings(images)
print('Encoding Complete')


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


     

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            x,y,w,h =faceLoc
            x,y,w,h =x*4,y*4,w*4,h*4
            
            if name in scr:
                face_region=[x + w,h + y]
                print(face_region)
                img[x:x + w+10,h:h + y-20,:]=cv2.blur(img[x:x + w+10, h:h + y-20,:],(30,30),cv2.BORDER_DEFAULT)
            else: 
                cv2.rectangle(img,(h,x),(y,w),(0,255,0),2)
                cv2.rectangle(img,(h,w-35),(y,w),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(h+6,w-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)

    
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
    
