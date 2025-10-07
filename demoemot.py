import cv2
from ultralytics import YOLO

model = YOLO("best.pt")


cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    results = model(source=frame,conf=0.5, stream=True)
    for r in results:
        annotated = r.plot()
        cv2.imshow("Emotion Detection", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()