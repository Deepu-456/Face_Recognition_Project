import cv2
import numpy as np
import face_recognition as fr
import os


path = 'Images'
image = []
className = []
List = os.listdir(path)
print(List)

for cl in List:
    curImg = cv2.imread(f'{path}/{cl}')
    image.append(curImg)
    className.append(os.path.splitext(cl)[0])
    print(className)


def Encodings(images):
    encodel = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodel.append(encode)
    return encodel





encodeListKnown = Encodings(image)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgsel = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgsel = cv2.cvtColor(imgsel, cv2.COLOR_BGR2RGB)

    facesCurFrame = fr.face_locations(imgsel)
    encodesCurFrame = fr.face_encodings(imgsel, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = fr.compare_faces(encodeListKnown, encodeFace)
        faceDis = fr.face_distance(encodeListKnown, encodeFace)
        # print(faceDistance)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            # print(name)
            a1, a2, b2, b1 = faceLoc
            b1, a2, b2, a1 = b1 * 4, a2 * 4, b2 * 4, a1 * 4
            cv2.rectangle(img, (a1, b1), (a2, b2), (0, 255, 0), 2)
            cv2.rectangle(img, (a1, b2 - 35), (a2, b2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (a1 + 6, b2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        if faceDis[matchIndex] < 0.50:
            name = className[matchIndex].upper()
        else:
            name = 'Unknown'
            # print(name)
            b1, a2, b2, a1 = faceLoc
            b1, a2, b2, a1 = b1 * 4, a2 * 4, b2 * 4, a1 * 4
            cv2.rectangle(img, (a1, b1), (a2, b2), (0, 255, 0), 2)
            cv2.rectangle(img, (a1, b2 - 35), (a2, b2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (a1 + 6, b2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
