import streamlit as st
import numpy as np
import os
import cv2
from face_recognition import train_dataset, recognize_Face_from_frame, create_dataset

# Constants
DATASET_DIR = "Dataset"
EMBEDDINGS_FILE = "embeddings.npy"

# App configuration
st.set_page_config(
    page_title="Face Recognition System",
    layout="centered",
    page_icon="🧠"
)

# Sidebar navigation
st.sidebar.title("Navigation")
menu_options = {
    "📸 Create Dataset": "dataset",
    "🧬 Train Model": "train",
    "🔍 Recognize Face": "recognize"
}
choice = st.sidebar.radio("Select Task", list(menu_options.keys()))

# 1️⃣ CREATE DATASET (Auto-capture)
if choice == "📸 Create Dataset":
    st.title("Create Face Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Person's Name", placeholder="Enter name")
    with col2:
        samples = st.slider("Samples to Capture", 10, 100, 30, 5)
    
    if st.button("🚀 Start Auto-Capture"):
        if not name.strip():
            st.error("Please enter a valid name")
        else:
            with st.spinner(f"🔍 Looking for faces to capture {samples} samples..."):
                try:
                    frames, saved_count = create_dataset(name, samples)
                    
                    if saved_count > 0:
                        st.success(f"✅ Successfully captured {saved_count} samples for {name}!")
                        st.balloons()
                        
                        # Show sample captures
                        st.subheader("Last 6 Captures")
                        cols = st.columns(3)
                        for i, frame in enumerate(frames[-6:]):
                            cols[i%3].image(frame, channels="BGR", use_column_width=True)
                    else:
                        st.warning("No faces detected during capture session")
                        
                except Exception as e:
                    st.error(f"Error during capture: {str(e)}")

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

# Add some app info
st.sidebar.markdown("---")
st.sidebar.info(
    "Face Recognition System\n\n"
    "1. Create Dataset: Capture face samples\n"
    "2. Train Model: Process all saved faces\n"
    "3. Recognize: Identify people in real-time"
)