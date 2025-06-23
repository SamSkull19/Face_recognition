import streamlit as st
import numpy as np
import os
import cv2  # Added missing import
from face_recognition import create_dataset, train_dataset, FaceRecognitionTransformer
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="Face Recognition System", layout="centered")
st.title("🧠 Real-time Face Recognition with DeepFace")

# Sidebar Menu
menu = ["📸 Create Dataset", "🧬 Train Model", "🔍 Recognize Face"]
choice = st.sidebar.selectbox("Select Task", menu)

# 1️⃣ CREATE DATASET (File upload version)
if choice == "📸 Create Dataset":
    name = st.text_input("Enter the name of the person:")
    uploaded_files = st.file_uploader(
        "Upload face images", 
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )
    
    if st.button("Create Dataset") and name.strip():
        if not uploaded_files:
            st.warning("Please upload at least one image")
        else:
            with st.spinner("Creating dataset..."):
                frames = create_dataset(name, uploaded_files)
                if frames:
                    st.success(f"Dataset created with {len(uploaded_files)} images for '{name}'")
                    st.subheader("Sample Images")
                    cols = st.columns(3)
                    for i, img in enumerate(frames[:3]):
                        cols[i].image(img, channels="BGR")
                else:
                    st.error("No valid faces detected in uploaded images")

# 2️⃣ TRAIN MODEL
elif choice == "🧬 Train Model":
    if st.button("Train Now"):
        with st.spinner("Training model..."):
            try:
                embeddings = train_dataset()
                np.save("embeddings.npy", embeddings)
                st.success("✅ Model trained successfully!")
                st.info(f"Learned {len(embeddings)} identities")
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

# 3️⃣ RECOGNIZE FACE (WebRTC version)
elif choice == "🔍 Recognize Face":
    if os.path.exists("embeddings.npy"):
        try:
            embeddings = np.load("embeddings.npy", allow_pickle=True).item()
            st.info("Starting camera... Please allow camera access when prompted")
            
            webrtc_ctx = webrtc_streamer(
                key="face-recognition",
                video_transformer_factory=lambda: FaceRecognitionTransformer(embeddings),
                async_transform=True,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "facingMode": "user"
                    },
                    "audio": False
                },
                rtc_configuration={  # Add this for HTTPS deployment
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }
            )
            
            if not webrtc_ctx.state.playing:
                st.warning("Waiting for camera access...")
                
        except Exception as e:
            st.error(f"Recognition error: {str(e)}")
    else:
        st.warning("⚠️ Please train the model first. No embeddings found.")

# Add troubleshooting info
st.sidebar.markdown("---")
st.sidebar.info(
    "ℹ️ Camera access requires:\n"
    "1. HTTPS connection\n"
    "2. Browser permission\n"
    "3. Working webcam"
)