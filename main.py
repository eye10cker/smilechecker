import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 1024)

face_cascade = cv2.CascadeClassifier('faces.xml')
smile_cascade = cv2.CascadeClassifier('smile.xml')

while cap.isOpened():
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=4)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=4)
    for (x1, y1, w1, h1) in faces:
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), color=(255, 0, 0), thickness=2)
        face_gray = gray[y1:y1 + h1, x1:x1 + w1]
        face_color = img[y1:y1 + h1, x1:x1 + w1]
        for (x, y, w, h) in smiles:
            cv2.rectangle(face_color, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    cv2.imshow('SmileChecker', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
