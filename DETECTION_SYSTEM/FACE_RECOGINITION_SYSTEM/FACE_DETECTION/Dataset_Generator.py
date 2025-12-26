import cv2
import os

def generate_dataset():
    # Path to Haar Cascade XML (already present in OpenCV)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_classifier = cv2.CascadeClassifier(cascade_path)

    # Function to detect and return cropped face
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            return None

        for (x, y, w, h) in faces:
            return img[y:y + h, x:x + w]  # return only first face

    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    id = 4
    img_id = 0

    # Create folder if not exists
    data_folder = os.path.join("DETECTION_SYSTEM", "FACE_RECOGINITION_SYSTEM", "data")
    os.makedirs(data_folder, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cropped = face_cropped(frame)

        if cropped is not None:
            img_id += 1
            face = cv2.resize(cropped, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = os.path.join(data_folder, f"user.{id}.{img_id}.jpg")
            cv2.imwrite(file_name_path, face)

            cv2.putText(face, str(img_id), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Cropped Face", face)

        if cv2.waitKey(1) == 13 or img_id == 1000:  # Press Enter key to stop
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Collecting samples is completed.")

# Run the function
generate_dataset()
