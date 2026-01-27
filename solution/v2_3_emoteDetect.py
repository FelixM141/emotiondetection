import os
import cv2
import numpy as np
from ultralytics import YOLO

import torch

MIN_FACE_W = 60
MIN_FACE_H = 60
MIN_FACE_AREA = 60*60


# Use mps backend on macOS with Apple silicon instead of cpu
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO("bestv2_3.pt")
model = model.to(device)

YUNET_PATH = "face_detection_yunet_2023mar.onnx"
assert os.path.exists(YUNET_PATH), f"YuNet-ONNX fehlt: {YUNET_PATH}"

# Init YuNet-Detector
def make_yunet(model_path, input_size=(320, 320), score_threshold=0.6, nms_threshold=0.3, top_k=5000):
    if hasattr(cv2, "FaceDetectorYN_create"):
        return cv2.FaceDetectorYN_create(
            model_path, "", input_size, score_threshold, nms_threshold, top_k,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
    else:
        return cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )

detector = make_yunet(YUNET_PATH, input_size=(320, 320))

# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")


FONT = cv2.FONT_HERSHEY_SIMPLEX
def clip(v, lo, hi): return max(lo, min(hi, v))

while True:
    ok, frame = cap.read()
    if not ok:
        print("No frame detected")
        break

    # optional Selfie-Flip:
    # frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # YuNet requires input size of current frame
    detector.setInputSize((w, h))

    # Face Recognition
    # returns: (retval, faces); faces: Nx15 (x, y, w, h, 10 Landmarks, score)
    #_, faces = detector.detect(frame)
    faces = detector.detect(frame)[1]

    if faces is not None:
        for f in faces:
            x, y, bw, bh = f[:4] # <float> values
            #score = float(f[-1])

            # Clipping (face should not be out of frame)
            x = clip(int(round(x)), 0, w - 1) # rounding because yunet works with subpixel coordinates
            y = clip(int(round(y)), 0, h - 1)
            bw = int(round(bw))
            bh = int(round(bh))

            if bw < MIN_FACE_W or bh < MIN_FACE_H:
                continue

            x2 = clip(x + bw, 0, w - 1)
            y2 = clip(y + bh, 0, h - 1)

            if x2 <= x or y2 <= y:
                continue

            # Padding
            pad = 0.1
            px = int(bw * pad); py = int(bh * pad)
            x0 = clip(x - px, 0, w - 1); y0 = clip(y - py, 0, h - 1)
            x1 = clip(x2 + px, 0, w - 1); y1 = clip(y2 + py, 0, h - 1)

            face_crop = frame[y0:y1, x0:x1]
            if face_crop.size == 0:
                continue

            # IMPORTANT: YOLO only on face_crop
            # results is list of YOLO results-objects with len 1 (r.orig_img, r.probs, r.names, ...)
            results = model.predict(source=face_crop, imgsz=320, verbose=False)

            if results and len(results) > 0:
                r = results[0]  # r is YOLO-result-object which incl. r.probs
                

                # YOLOv8-cls: probability in r.probs
                if hasattr(r, "probs") and r.probs is not None:
                    probs = r.probs.data.clone()  # Tensor mit allen Klassen-Wahrscheinlichkeiten

                    # Bias
                    HAPPY_ID = 2
                    SAD_ID = 4
                    probs[HAPPY_ID] *= 1.2
                    probs[SAD_ID] *= 1.1

                    # Neue Top-Klasse bestimmen
                    cls_id = int(torch.argmax(probs))
                    conf = float(probs[cls_id])
                    label = model.names[cls_id]


                # Display
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x0, max(0, y0 - 8)),
                            FONT, 0.7, (0, 255, 0), 2)


    cv2.imshow("Emotion Detection (YuNet + YOLO)", frame)
    if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
