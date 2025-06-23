import os
import cv2
import numpy as np
from deepface import DeepFace
from streamlit_webrtc import VideoTransformerBase

# Path to dataset directory
dir = "Dataset"
os.makedirs(dir, exist_ok=True)

# ✅ Create Dataset Function (now uses file upload)
def create_dataset(name, uploaded_files):
    person = os.path.join(dir, name)
    os.makedirs(person, exist_ok=True)
    frames = []

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") \
                .detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                face_path = os.path.join(person, f"{name}_{i+1}.jpg")
                cv2.imwrite(face_path, face_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            frames.append(frame)
        except Exception as e:
            print(f"Error processing image {i+1}: {str(e)}")

    return frames

# ✅ Train Dataset Function (unchanged)
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

# ✅ Modified Recognize Face Function for single frames
def recognize_face(frame, embeddings):
    try:
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
                           0.5, (255, 255, 255), 2)

            except Exception as e:
                print("Face analysis error:", str(e))

    except Exception as e:
        print("Frame processing error:", str(e))

    return frame

# WebRTC Video Transformer Class
class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return recognize_face(img, self.embeddings)