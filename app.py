import streamlit as st
import numpy as np
import os
import cv2
import time
from face_recognition import train_dataset, recognize_Face_from_frame

# Constants
DATASET_DIR = "Dataset"
EMBEDDINGS_FILE = "embeddings.npy"

# App configuration
st.set_page_config(
    page_title="Face Recognition System",
    layout="centered",
    page_icon="🧠"
)

# Suppress OpenCV and TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# Sidebar navigation
st.sidebar.title("Navigation")
menu_options = {
    "📸 Create Dataset": "dataset",
    "🧬 Train Model": "train",
    "🔍 Recognize Face": "recognize"
}
choice = st.sidebar.radio("Select Task", list(menu_options.keys()))

# Helper function for camera capture
def capture_images(name, sample_count):
    """Handles the image capture process"""
    if 'capture_count' not in st.session_state:
        st.session_state.capture_count = 0
        st.session_state.captured_frames = []
    
    placeholder = st.empty()
    img_placeholder = st.empty()
    status = st.empty()
    
    while st.session_state.capture_count < sample_count:
        with placeholder.container():
            st.info(f"Auto-capturing {st.session_state.capture_count + 1}/{sample_count} - Look at the camera")
        
        img_file_buffer = st.camera_input("", key=f"camera_{st.session_state.capture_count}")
        
        if img_file_buffer is not None:
            # Process the image
            bytes_data = img_file_buffer.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Save the image
            person_dir = os.path.join(DATASET_DIR, name)
            os.makedirs(person_dir, exist_ok=True)
            st.session_state.capture_count += 1
            cv2.imwrite(os.path.join(person_dir, f"{name}_{st.session_state.capture_count}.jpg"), frame)
            st.session_state.captured_frames.append(frame)
            
            # Show preview
            with img_placeholder.container():
                st.image(frame, channels="BGR", caption=f"Captured #{st.session_state.capture_count}")
            
            # Small delay before next capture
            time.sleep(1)
            
            # Force rerun to refresh camera input
            st.rerun()
    
    if st.session_state.capture_count >= sample_count:
        with status.container():
            st.success(f"✅ Successfully captured {sample_count} images for '{name}'!")
            st.balloons()
            
            # Show sample captures
            st.subheader("Sample Captures")
            cols = st.columns(3)
            for i, frame in enumerate(st.session_state.captured_frames[-6:]):
                cols[i%3].image(frame, channels="BGR", use_column_width=True)
        
        # Reset session state
        del st.session_state.capture_count
        del st.session_state.captured_frames
        return True
    return False

# 1️⃣ CREATE DATASET
if choice == "📸 Create Dataset":
    st.title("Create Face Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Person's Name", placeholder="Enter name")
    with col2:
        samples = st.slider("Number of Images to Capture", 5, 100, 20, 5)
    
    if name.strip() and st.button("🚀 Start Auto-Capture"):
        with st.spinner("Initializing camera..."):
            capture_images(name, samples)

# 2️⃣ TRAIN MODEL
elif choice == "🧬 Train Model":
    st.title("Train Recognition Model")
    
    if st.button("🔄 Train Model Now"):
        if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
            st.error("No dataset found. Please create a dataset first.")
        else:
            with st.spinner("Training model (this may take a while)..."):
                try:
                    embeddings = train_dataset()
                    np.save(EMBEDDINGS_FILE, embeddings)
                    st.success("Model trained successfully!")
                    st.json({"People in dataset": list(embeddings.keys())})
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

# 3️⃣ RECOGNIZE FACES
elif choice == "🔍 Recognize Face":
    st.title("Real-time Face Recognition")
    
    if not os.path.exists(EMBEDDINGS_FILE):
        st.error("Please train the model first")
    else:
        embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        
        st.info("Point your face at the camera for recognition")
        img_file = st.camera_input("Real-time Recognition")
        
        if img_file is not None:
            bytes_data = img_file.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            with st.spinner("Analyzing face..."):
                processed_frame = recognize_Face_from_frame(frame, embeddings)
                st.image(processed_frame, channels="BGR", use_column_width=True)

# App information
st.sidebar.markdown("---")
st.sidebar.info(
    "Face Recognition System\n\n"
    "1. Create Dataset: Capture face samples\n"
    "2. Train Model: Process all saved faces\n"
    "3. Recognize: Identify people in real-time"
)