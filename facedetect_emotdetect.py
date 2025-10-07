import cv2
from ultralytics import YOLO

# 🧠 YOLOv8 Emotionserkennungsmodell laden
model = YOLO("best.pt")

# 🧠 Haarcascade für Gesichtserkennung laden
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 📷 Kamera starten
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # 🔍 Gesicht erkennen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # ✂️ Gesicht ausschneiden
        face_crop = frame[y:y+h, x:x+w]

        # 🧠 YOLO auf das Gesicht anwenden
        results = model(source=face_crop, conf=0.5, stream=True)

        for r in results:
            annotated = r.plot()
            cv2.imshow("Emotion Detection", annotated)

    # 🛑 Mit ESC beenden
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
