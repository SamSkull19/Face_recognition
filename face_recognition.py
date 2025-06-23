import os
import cv2
import numpy as np
from deepface import DeepFace
import time

# Path to dataset directory
DATASET_DIR = "Dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

def create_dataset(name, sample_count=50):
    """
    Automatically captures face samples at 1-second intervals when faces are detected
    Returns: List of captured frames and count of saved images
    """
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam")
    
    count = 0
    frames = []
    last_capture_time = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    try:
        while count < sample_count:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            current_time = time.time()
            if len(faces) > 0 and (current_time - last_capture_time) > 1.0:  # 1 second interval
                for (x, y, w, h) in faces:
                    count += 1
                    face_img = frame[y:y+h, x:x+w]
                    cv2.imwrite(os.path.join(person_dir, f"{name}_{count}.jpg"), face_img)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    last_capture_time = current_time
                    frames.append(frame.copy())
                    break  # Only capture one face per interval
            
            # Show preview (not needed for Streamlit but useful for debugging)
            cv2.imshow('Capturing Faces - Press Q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return frames, count

def train_dataset():
    """Trains the model using saved face images"""
    embeddings = {}
    
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if os.path.isdir(person_dir):
            embeddings[person_name] = []
            
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                try:
                    result = DeepFace.represent(
                        img_path=img_path,
                        model_name="Facenet",
                        enforce_detection=False
                    )
                    if result:
                        embeddings[person_name].append(result[0]["embedding"])
                except Exception as e:
                    print(f"Error processing {img_name}: {str(e)}")
    
    return embeddings

def recognize_Face_from_frame(frame, embeddings):
    """Recognizes faces in a single frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        try:
            # Get face attributes
            analysis = DeepFace.analyze(face_img, actions=["age", "gender", "emotion"], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            age = analysis["age"]
            gender = max(analysis["gender"], key=analysis["gender"].get)
            emotion = max(analysis["emotion"], key=analysis["emotion"].get)
            
            # Get embedding and compare
            face_embedding = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]
            
            # Find best match
            match, max_similarity = None, -1
            for person_name, person_embeddings in embeddings.items():
                for embed in person_embeddings:
                    similarity = np.dot(face_embedding, embed) / (
                        np.linalg.norm(face_embedding) * np.linalg.norm(embed)
                    )
                    if similarity > max_similarity:
                        max_similarity = similarity
                        match = person_name

            label = f"{match} ({max_similarity:.2f})" if max_similarity > 0.7 else "Unknown"
            cv2.putText(
                frame,
                f"{label} | Age: {int(age)} | {gender} | {emotion}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )
            
        except Exception as e:
            print(f"Face analysis error: {str(e)}")
    
    return frame