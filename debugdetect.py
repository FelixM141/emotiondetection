import cv2
import time
from ultralytics import YOLO

# YOLOv8 Emotionserkennungsmodell laden
model = YOLO("best.pt")

# Haarcascade für Gesichtserkennung laden
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Kamera starten
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # --- Haarcascade-Zeitmessung ---
    t0 = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    t1 = time.perf_counter()
    haar_time = (t1 - t0) * 1000  # in Millisekunden

    # --- YOLO-Zeitmessung ---
    yolo_time = 0.0
    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]

        t2 = time.perf_counter()
        results = model(source=face_crop, conf=0.5, stream=True)
        t3 = time.perf_counter()

        yolo_time += (t3 - t2) * 1000  # kumuliert für alle Gesichter

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                cv2.putText(frame, f"{label} ({conf:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Laufzeiten anzeigen
    cv2.putText(frame, f"Haarcascade: {haar_time:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"YOLO: {yolo_time:.1f} ms", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Emotion Detection (mit Zeitmessung)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
