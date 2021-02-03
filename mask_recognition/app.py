import os
import cv2 as cv
import numpy as np

DIR = r'data'
cascade_dir = 'haar_face.xml'

haar_cascade = cv.CascadeClassifier('haar_face.xml')


def create_dataset(DIR, cascade_dir):
    group = []
    for i in os.listdir(DIR):
        group.append(i)

    haar_cascade = cv.CascadeClassifier(cascade_dir)

    features = []
    labels = []

    for item in group:
        path = os.path.join(DIR, item)
        label = group.index(item)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=1)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

    features = np.array(features, dtype='object')
    labels = np.array(labels)

    return features, labels, group


features, labels, group = create_dataset(DIR, cascade_dir)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

webcam = cv.VideoCapture(1)

while True:
    successful_frame_read_read, frame = webcam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_coordinates = haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in face_coordinates:
        faces_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(faces_roi)
        cv.putText(frame, str(group[label]), (x, y-20),
                   cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), thickness=2)
        cv.putText(frame, str(round(confidence, 2)) + ' %', (x+120, y-20),
                   cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), thickness=2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.imshow('webcam', frame)
    key = cv.waitKey(1)
    if key == 81 or key == 113:
        break

webcam.release()
