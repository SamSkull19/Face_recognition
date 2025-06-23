import streamlit as st
import numpy as np
import os
from face_recognition import create_dataset, train_dataset, recognize_Face

st.set_page_config(page_title="Face Recognition System", layout="centered")
st.title("🧠 Real-time Face Recognition with DeepFace")

# Sidebar Menu
menu = ["📸 Create Dataset", "🧬 Train Model", "🔍 Recognize Face"]
choice = st.sidebar.selectbox("Select Task", menu)

# 1️⃣ CREATE DATASET
if choice == "📸 Create Dataset":
    name = st.text_input("Enter the name of the person:")
    samples = st.slider("Number of face samples", 10, 100, 50, 10)

    if st.button("Start Capturing"):
        if name.strip() == "":
            st.warning("⚠️ Please enter a valid name.")
        else:
            st.info("📷 Starting webcam. Look at the camera...")
            frames = create_dataset(name, sample_count=samples)
            st.success(f"✅ Dataset created with {samples} images for '{name}'")

            # Show last 5 captured faces
            st.subheader("📷 Captured Samples")
            for f in frames[-5:]:
                st.image(f, channels="BGR")

# 2️⃣ TRAIN MODEL
elif choice == "🧬 Train Model":
    if st.button("Train Now"):
        st.info("🔄 Training embeddings from dataset...")
        embeddings = train_dataset()
        np.save("embeddings.npy", embeddings)
        st.success("✅ Model trained and embeddings saved.")

# 3️⃣ RECOGNIZE FACE
elif choice == "🔍 Recognize Face":
    if os.path.exists("embeddings.npy"):
        if st.button("Start Recognition"):
            st.info("🎥 Starting webcam for recognition...")
            embeddings = np.load("embeddings.npy", allow_pickle=True).item()
            frames = recognize_Face(embeddings)

            st.subheader("🔍 Recognition Result")
            for f in frames:
                st.image(f, channels="BGR")
    else:
        st.warning("⚠️ Please train the model first. No embeddings found.")


