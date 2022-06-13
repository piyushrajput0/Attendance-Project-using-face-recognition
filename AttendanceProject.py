import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Image'
print("Encoding all the Images . . .")
images = []
classNames = []
myList = os.listdir(path)
# print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')    # cls is the name of our images
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# print(classNames)


def findencodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('AttendanceFile.csv', 'r+') as attend:
        myDataList = attend.readlines()
        # print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            # print(now)
            dtString = now.strftime('%H:%M:%S')
            attend.writelines(f'\n{name},{dtString}')
            print(" Enter : ", name,)


# markAttendance('Piyush')


encodeListKnown = findencodings(images)
print(' \n \n Encoding of Images Complete. . . ')
print("\n", "\n", "Webcam of your device is going to be switch on")

cap = cv2.VideoCapture(0)  # SETTING UP YOUR WEB CAM 0 is as a id

while True:  # while true for each frame
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)   # reduce the size of img to enhance
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facescurframe = face_recognition.face_locations(imgS)   # face_locations
    # it is possible there are a lot of faces in the webcam, so it finds us the locations of
    # images, and then it sends to me

    encodescurframe = face_recognition.face_encodings(imgS, facescurframe)

    # we go through the all the faces that we found in curframe and search in our images

    for encodeface, faceloc in zip(encodescurframe, facescurframe):

        # it find facelocation from facescurrframe and then its encoding from
        # encodecurrframe

        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        # it returns list  by comparing with faces with available one
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)


        cv2.imshow('Webcam', img)
        cv2.waitKey(1)


