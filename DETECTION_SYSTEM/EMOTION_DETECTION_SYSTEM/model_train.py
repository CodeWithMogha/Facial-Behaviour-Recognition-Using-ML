import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import mediapipe as mp

# ======== Load dataset from directories ========
def load_dataset(base_path):
    images = []
    labels = []
    emotions = sorted(os.listdir(base_path))  # e.g. ['angry', 'happy', 'sad', ...]

    for emotion in emotions:
        emotion_folder = os.path.join(base_path, emotion)
        if not os.path.isdir(emotion_folder):
            continue
        for file in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (48, 48))
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(emotion)

    images = np.array(images)
    images = np.expand_dims(images, -1)  # Shape: (num_samples, 48, 48, 1)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_one_hot = to_categorical(labels_encoded)

    return images, labels_one_hot, le.classes_

# ======== Build CNN Model ========
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotion classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ======== Paths to your data folders ========
train_path = r'FER-2013 DATASET/train'
test_path = r'FER-2013 DATASET/test'

# ======== Load data ========
print("Loading training data...")
x_train, y_train, emotion_labels = load_dataset(train_path)
print("Loading testing data...")
x_test, y_test, _ = load_dataset(test_path)

print(f"Train shape: {x_train.shape}, Train labels shape: {y_train.shape}")
print(f"Test shape: {x_test.shape}, Test labels shape: {y_test.shape}")
print("Emotion labels:", emotion_labels)

# ======== Build and train the model ========
model = build_model()
model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
model.save('emotion_recognition_model.h5')
print("Model saved to disk.")

# ======== Real-time Emotion Detection from Webcam ========
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(1)
print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

            # Handle bounding box overflow
            x = max(0, x)
            y = max(0, y)
            bw = min(w - x, bw)
            bh = min(h - y, bh)

            face = frame[y:y+bh, x:x+bw]
            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized.astype('float32') / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))

            prediction = model.predict(reshaped)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)

    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 