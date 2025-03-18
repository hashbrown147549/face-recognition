import cv2 as cv
import numpy as np

people = ["you","me"]

# features = np.load('features.npy')
# labels = np.load('labels.npy')
haar_cascade = cv.CascadeClassifier('haar_face.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

capture = cv.VideoCapture(0)
while True:
    isTrue, img = capture.read()
    

    gray = cv.cvtColor(img,cv.COLOR_BAYER_BGR2GRAY)
    cv.imshow('Person',gray)

    faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+h]
        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with confidence of {confidence}')
        cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow('detected face',img)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
