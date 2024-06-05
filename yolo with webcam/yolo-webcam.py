from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


model = YOLO("../Yolo-Weights/yolov8l.pt")

while True:
    success, img = cap.read()

    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    w, h = x2 - x1, y2 - y1
    cvzone.cornerRect(img, (x1, y1, w, h))
    conf = math.ceil((box.conf[0] * 100)) / 100
    print(conf)
    cvzone.putTextRect(img,f'{conf}',(x1,y1-20))

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
