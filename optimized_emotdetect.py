import cv2
from ultralytics import YOLO

# YOLOv8 Emotionserkennungsmodell laden
model = YOLO("best.pt")

# Haarcascade für Gesichtserkennung laden
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Kamera starten
cap = cv2.VideoCapture(0)
cv2.setUseOptimized(True)

frame_count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Frame verkleinern für weniger CPU-Last (optional)
    frame = cv2.resize(frame, (640, 480))

    # Gesichtserkennung
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Nur größtes Gesicht verwenden (optional)
    if len(faces) > 0:
        faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
        x, y, w, h = faces[0]

        # Nur alle 5 Frames YOLO anwenden
        if frame_count % 5 == 0:
            face_crop = frame[y:y+h, x:x+w]
            results = model(source=face_crop, conf=0.5, stream=True)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]

                    # Emotion ins Originalbild einzeichnen
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Ganzes Bild anzeigen
    cv2.imshow("Emotion Detection", frame)
    frame_count += 1

    # ESC zum Beenden
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
