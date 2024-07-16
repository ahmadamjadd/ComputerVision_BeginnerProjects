#pylint:disable=no-member

import cv2 as cv

people = ['Ahmad', 'Eshaal', 'Hareem', 'Minahil']

haar_cascade = cv.CascadeClassifier('haar_face.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("You're very beautiful, We cannot open your video")
    exit

while True:
    r, frame =  cap.read()
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        cv.putText(frame, str(people[label]), (x,y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    

    if r == True:
        cv.imshow("Camera", frame)
        cv.imshow("Gray_Scaled", gray)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()