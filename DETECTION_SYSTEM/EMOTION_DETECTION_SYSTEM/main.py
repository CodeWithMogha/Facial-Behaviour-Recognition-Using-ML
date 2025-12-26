import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load your model once
model = load_model('emotion_recognition_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def predict_emotion_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

            x, y = max(0, x), max(0, y)
            bw, bh = min(w - x, bw), min(h - y, bh)

            face = frame[y:y+bh, x:x+bw]
            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))

            prediction = model.predict(reshaped, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            return emotion
    return "No Face"

