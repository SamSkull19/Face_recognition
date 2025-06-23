import os
import cv2
import numpy as np
from deepface import DeepFace

# Path to dataset directory
dir = "Dataset"
os.makedirs(dir, exist_ok=True)


# ✅ Create Dataset Function
def create_dataset(name, sample_count=50):
    person = os.path.join(dir, name)
    os.makedirs(person, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    frames = []  # to return for streamlit

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") \
            .detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y + h, x:x + w]
            face_path = os.path.join(person, f"{name}_{count}.jpg")
            cv2.imwrite(face_path, face_img)

            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frames.append(frame.copy())

        if count >= sample_count:
            break

    cap.release()
    return frames


# ✅ Train Dataset Function
def train_dataset():
    embedding = {}

    for i in os.listdir(dir):
        person = os.path.join(dir, i)
        if os.path.isdir(person):
            embedding[i] = []
            for img_name in os.listdir(person):
                img_path = os.path.join(person, img_name)
                try:
                    vec = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    embedding[i].append(vec)
                except Exception as e:
                    print("Failed to process image:", img_name)

    return embedding


# ✅ Recognize Face Function
def recognize_Face(embeddings):
    cap = cv2.VideoCapture(0)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") \
            .detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            try:
                analyse = DeepFace.analyze(face_img, actions=["age", "gender", "emotion"], enforce_detection=False)
                if isinstance(analyse, list):
                    analyse = analyse[0]

                age = analyse["age"]
                gender = analyse["gender"]
                gender = gender if isinstance(gender, str) else max(gender, key=gender.get)
                emotion = max(analyse["emotion"], key=analyse["emotion"].get)

                face_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]

                match = None
                max_similarity = -1

                for person_name, person_embeddings in embeddings.items():
                    for embed in person_embeddings:
                        similarity = np.dot(face_embedding, embed) / (
                            np.linalg.norm(face_embedding) * np.linalg.norm(embed)
                        )
                        if similarity > max_similarity:
                            max_similarity = similarity
                            match = person_name

                if max_similarity > 0.7:
                    label = f"{match} ({max_similarity:.2f})"
                else:
                    label = "Unknown"

                display_text = f"{label} | Age: {int(age)} | Gender: {gender} | Emotion: {emotion}"
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255), thickness=2)

            except Exception as e:
                print("Face could not be analyzed.")

        frames.append(frame.copy())
        break  # Only take one frame for Streamlit real-time preview

    cap.release()
    return frames
