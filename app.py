import streamlit as st
import numpy as np
import os
import cv2
from face_recognition import train_dataset, recognize_Face_from_frame

# Streamlit page config
st.set_page_config(page_title="Face Recognition System", layout="centered")
st.title("🧠 Real-time Face Recognition with DeepFace")

# Sidebar Menu
menu = ["📸 Create Dataset", "🧬 Train Model", "🔍 Recognize Face"]
choice = st.sidebar.selectbox("Select Task", menu)

# 1️⃣ CREATE DATASET (Auto Capture Images)
if choice == "📸 Create Dataset":
    name = st.text_input("Enter the name of the person:")
    samples = st.slider("Number of face samples", 5, 100, 10, 5)

    if name.strip() == "":
        st.warning("⚠️ Please enter a valid name.")
    else:
        person_dir = os.path.join("Dataset", name)
        os.makedirs(person_dir, exist_ok=True)

        if "start_capture" not in st.session_state:
            st.session_state.start_capture = False
        if "capture_count" not in st.session_state:
            st.session_state.capture_count = len(os.listdir(person_dir))

        if not st.session_state.start_capture:
            if st.button("📸 Start Capturing"):
                st.session_state.start_capture = True
                st.experimental_rerun()

        if st.session_state.start_capture:
            st.info(f"📷 Capturing image {st.session_state.capture_count + 1} of {samples}")

            img_file_buffer = st.camera_input("Keep looking at the camera", key=f"cam_{st.session_state.capture_count}")

            if img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                img_name = os.path.join(person_dir, f"{name}_{st.session_state.capture_count + 1}.jpg")
                cv2.imwrite(img_name, image)

                st.success(f"✅ Saved image {st.session_state.capture_count + 1}")
                st.session_state.capture_count += 1

                if st.session_state.capture_count >= samples:
                    st.success("🎉 Dataset capture complete!")
                    st.balloons()
                    st.session_state.start_capture = False
                    st.session_state.capture_count = 0
                else:
                    st.experimental_rerun()


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
        embeddings = np.load("embeddings.npy", allow_pickle=True).item()
        st.info("🎥 Look at the camera for recognition")

        img_file_buffer = st.camera_input("Take a picture for recognition", key="camera_recognize")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            processed_frame = recognize_Face_from_frame(frame, embeddings)
            st.image(processed_frame, channels="BGR")
    else:
        st.warning("⚠️ Please train the model first. No embeddings found.")
