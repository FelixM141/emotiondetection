# emotiondetection
Emotion Detection using Yunet as face detection, and trained with Roboflow dataset for Emotion Detection.

# Dataset
The pictures that were had both colored and gray pictures, to find the exact same pictures, go to:
* https://universe.roboflow.com/pea-edu/emotion-recognition-5ohyf
    * All training, validation and test images were cropped to 320x320 pixels.



# Features
* Live Camera using the standard camera (0)
* Face detection, with ONNX Yunet Model.
* Padding to avoid cutting faces.
* Emotion classification on cropped area, using self trained yolov8 model.
* Displayed information on predicted emotion (confidence score).
* Bounding boxes around the detected face as visualization.


# Requirements
To install all of the dependencies, it is recommended to open a Venv and use !pip install -r requirements.txt

#