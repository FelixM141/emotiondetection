import cv2
from ultralytics import YOLO

# YOLOv8 Emotionserkennungsmodell laden
model = YOLO("best.pt")

# YuNet-Modellpfad (wird mit OpenCV geliefert)
model_path = cv2.data.haarcascades.replace("haarcascades/", "") + "face_detection_yunet_2023mar.onnx"

# YuNet-Detektor initialisieren
detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(320, 320),   # sollte später dynamisch angepasst werden
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# Kamera starten
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Bildgröße anpassen (YuNet braucht Inputgröße)
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    # Gesichter erkennen mit YuNet
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])

            # Gesicht ausschneiden
            face_crop = frame[y:y + h, x:x + w]

            # YOLO Emotionserkennung auf Gesicht anwenden
            results = model(source=face_crop, conf=0.5, stream=True)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]

                    # Text & Rahmen zeichnen
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Emotion Detection (YuNet)", frame)

    # ESC zum Beenden
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
