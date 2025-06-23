import streamlit as st
import numpy as np
import os
import cv2
from face_recognition import create_dataset, train_dataset, FaceRecognitionTransformer, recognize_face
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# Set page config
st.set_page_config(
    page_title="Face Recognition System",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve visuals
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stAlert {
        padding: 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Real-time Face Recognition with DeepFace")

# Sidebar Menu
with st.sidebar:
    st.header("Navigation")
    menu = st.radio("Select Task", 
                   ["📸 Create Dataset", "🧬 Train Model", "🔍 Recognize Face"],
                   label_visibility="collapsed")

# 1️⃣ CREATE DATASET (File upload version)
if menu == "📸 Create Dataset":
    st.subheader("Create New Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Person's Name", placeholder="Enter name")
        samples = st.slider("Target Samples", 10, 100, 20)
    
    with col2:
        uploaded_files = st.file_uploader(
            "Upload Face Images",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            help="Upload multiple images of the same person"
        )

    if st.button("🚀 Create Dataset", type="primary") and name.strip():
        if not uploaded_files:
            st.warning("Please upload at least one image")
        else:
            with st.spinner(f"Processing {len(uploaded_files)} images..."):
                try:
                    frames = create_dataset(name, uploaded_files, sample_count=samples)
                    st.success(f"✅ Success! Created dataset for '{name}' with {len(frames)} samples")
                    
                    st.subheader("Sample Preview")
                    cols = st.columns(4)
                    for idx, img in enumerate(frames[:4]):
                        cols[idx%4].image(img, channels="BGR", use_column_width=True)
                except Exception as e:
                    st.error(f"Error creating dataset: {str(e)}")

# 2️⃣ TRAIN MODEL
elif menu == "🧬 Train Model":
    st.subheader("Train Recognition Model")
    
    if not os.path.exists("Dataset") or len(os.listdir("Dataset")) == 0:
        st.warning("No dataset found. Please create a dataset first.")
    else:
        if st.button("🔧 Train Model", type="primary"):
            with st.spinner("Training model. This may take a few minutes..."):
                try:
                    progress_bar = st.progress(0)
                    embeddings = train_dataset()
                    np.save("embeddings.npy", embeddings)
                    progress_bar.progress(100)
                    st.success("🎉 Model trained successfully!")
                    st.info(f"Learned {len(embeddings)} distinct faces")
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

# 3️⃣ RECOGNIZE FACE (WebRTC with fallback)
elif menu == "🔍 Recognize Face":
    st.subheader("Real-time Recognition")
    
    if not os.path.exists("embeddings.npy"):
        st.warning("Please train the model first")
    else:
        try:
            embeddings = np.load("embeddings.npy", allow_pickle=True).item()
            
            # WebRTC Configuration
            rtc_config = RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]}
                ]
            })
            
            st.info("Starting camera... Please allow camera access when prompted")
            
            # WebRTC Streamer
            ctx = webrtc_streamer(
                key="face-recognition",
                video_transformer_factory=lambda: FaceRecognitionTransformer(embeddings),
                rtc_configuration=rtc_config,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "facingMode": "user"
                    },
                    "audio": False
                },
                async_transform=True
            )
            
            # Fallback to file upload if WebRTC fails
            if not ctx.state.playing:
                st.warning("Camera not accessible. Try file upload instead:")
                uploaded_file = st.file_uploader(
                    "Upload Image for Recognition",
                    type=["jpg", "png", "jpeg"],
                    key="recognition_upload"
                )
                
                if uploaded_file is not None:
                    with st.spinner("Processing image..."):
                        try:
                            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                            result = recognize_face(img, embeddings)
                            st.image(result, channels="BGR", caption="Recognition Result")
                        except Exception as e:
                            st.error(f"Recognition error: {str(e)}")
        
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")

# Add troubleshooting info
with st.sidebar:
    st.markdown("---")
    st.subheader("Troubleshooting")
    st.markdown("""
    - **Camera not working?**  
      • Refresh the page  
      • Check browser permissions  
      • Try Chrome/Firefox  
      
    - **Slow performance?**  
      • Reduce image size  
      • Close other tabs  
      
    - **Deployment issues?**  
      • Ensure HTTPS is enabled  
      • Check network firewall  
    """)

# Version info
st.sidebar.markdown("---")
st.sidebar.caption("v2.1 | WebRTC Enhanced Edition")