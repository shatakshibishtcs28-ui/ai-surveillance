"""Restricted zone mouse se chunne ke liye.
Run: python pick_zone.py            (webcam)
     python pick_zone.py video.mp4  (video)
Frame pe 4 corners click karo. Enter = save, r = reset, q = cancel."""
import json
import sys

import cv2

src = sys.argv[1] if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(src)
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit("Video/webcam khul nahi paya.")

pts = []


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y])


cv2.namedWindow("Pick zone")
cv2.setMouseCallback("Pick zone", on_click)
while True:
    img = frame.copy()
    for p in pts:
        cv2.circle(img, tuple(p), 5, (0, 0, 255), -1)
    if len(pts) > 1:
        cv2.polylines(img, [__import__("numpy").array(pts)], len(pts) == 4,
                      (0, 0, 255), 2)
    cv2.putText(img, "Click 4 corners | Enter=save r=reset q=cancel", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Pick zone", img)
    k = cv2.waitKey(30) & 0xFF
    if k == ord("r"):
        pts.clear()
    elif k == ord("q"):
        break
    elif k == 13 and len(pts) == 4:
        json.dump(pts, open("zone.json", "w"))
        print("Zone saved in zone.json:", pts)
        break
cv2.destroyAllWindows()
