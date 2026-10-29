"""Test: YOLO webcam me kya-kya dekh raha hai (sab objects, low confidence).
q = band"""
import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model(frame, conf=0.10, verbose=False)[0]
    cv2.imshow("TEST - sab objects (q = band)", r.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
