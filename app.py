import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from collections import deque

# --- Page Configuration ---
st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 10px 0;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-low {
        color: #FF9800;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# --- Load Model Function ---
@st.cache_resource
def load_asl_model(model_path, classes_path):
    """Load the trained model and class names"""
    try:
        model = load_model(model_path)
        with open(classes_path, "rb") as f:
            class_names = pickle.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


# --- Extract Landmarks ---
def extract_landmarks_from_frame(hand_landmarks):
    """Extract 63 features from detected hand"""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords)


# --- Video Processor Class ---
class ASLVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.class_names = None
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.prediction_history = deque(maxlen=5)
        self.current_prediction = "NOTHING"
        self.current_confidence = 0.0
        self.top3_predictions = []

    def set_model(self, model, class_names):
        """Set the model and class names"""
        self.model = model
        self.class_names = class_names

    def recv(self, frame):
        """Process each frame from the webcam"""
        img = frame.to_ndarray(format="bgr24")
        
        if self.model is None:
            # Display "Load Model First" message
            cv2.putText(
                img,
                "Load Model First!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Flip for mirror effect
        img = cv2.flip(img, 1)
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        predicted_class = "NOTHING"
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

                # Extract landmarks for prediction
                landmarks = extract_landmarks_from_frame(hand_landmarks)
                landmarks = landmarks.reshape(1, -1)

                # Predict
                predictions = self.model.predict(landmarks, verbose=0)
                predicted_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_idx]

                # Smooth predictions
                self.prediction_history.append(predicted_idx)
                
                # Use most common prediction
                if len(self.prediction_history) > 0:
                    final_prediction = max(
                        set(self.prediction_history), key=list(self.prediction_history).count
                    )
                    predicted_class = self.class_names[final_prediction]

                # Get top 3 predictions
                top3_indices = np.argsort(predictions[0])[-3:][::-1]
                self.top3_predictions = [
                    (self.class_names[idx], predictions[0][idx]) for idx in top3_indices
                ]

                # Update current prediction
                self.current_prediction = predicted_class
                self.current_confidence = confidence

                # Display prediction on frame
                text = f"{predicted_class}: {confidence*100:.1f}%"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                
                # Background rectangle
                cv2.rectangle(
                    img, (10, 10), (20 + text_size[0], 60), (0, 0, 0), -1
                )

                # Text color based on confidence
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                cv2.putText(
                    img, text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
                )
        else:
            # No hand detected
            cv2.putText(
                img,
                "Show your hand",
                (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3,
            )
            self.prediction_history.clear()
            self.current_prediction = "NOTHING"
            self.current_confidence = 0.0
            self.top3_predictions = []

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Main App ---
def main():
    # Header
    st.markdown(
        '<p class="main-header">🤟 Real-Time ASL Recognition</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Live American Sign Language Recognition using AI</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model paths
        st.subheader("📁 Model Configuration")
        model_path = st.text_input(
            "Model Path",
            value="models/asl_landmarks_final.h5",
            help="Path to the trained .h5 model file",
        )

        classes_path = st.text_input(
            "Classes Path",
            value="models/asl_landmarks_classes.pkl",
            help="Path to the classes pickle file",
        )

        # Load model button
        load_model_btn = st.button("🔄 Load Model", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown(
            """
        1. Click **Load Model** above
        2. Click **START** on the video
        3. Allow camera access in your browser
        4. Show ASL hand signs to the camera
        5. View real-time predictions below
        """
        )

        st.markdown("---")
        st.info("💡 **Tip:** Make sure your hand is clearly visible and well-lit!")

    # Initialize session state
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "model" not in st.session_state:
        st.session_state.model = None
    if "class_names" not in st.session_state:
        st.session_state.class_names = None

    # Load model
    if load_model_btn:
        with st.spinner("Loading model..."):
            model, class_names = load_asl_model(model_path, classes_path)
            if model is not None:
                st.session_state.model = model
                st.session_state.class_names = class_names
                st.session_state.model_loaded = True
                st.sidebar.success(f"✅ Model loaded! Classes: {len(class_names)}")
            else:
                st.sidebar.error("❌ Failed to load model. Check the file paths.")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Camera Feed")
        
        # Create video processor
        ctx = webrtc_streamer(
            key="asl-recognition",
            video_processor_factory=ASLVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Set model in processor if loaded
        if st.session_state.model_loaded and ctx.video_processor:
            ctx.video_processor.set_model(
                st.session_state.model, st.session_state.class_names
            )

    with col2:
        st.subheader("📊 Live Predictions")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load the model first!")
        else:
            # Placeholder for predictions
            prediction_placeholder = st.empty()
            top3_placeholder = st.empty()
            
            # Display predictions from processor
            if ctx.video_processor:
                if ctx.video_processor.current_prediction != "NOTHING":
                    conf_class = (
                        "confidence-high"
                        if ctx.video_processor.current_confidence > 0.7
                        else "confidence-low"
                    )
                    prediction_placeholder.markdown(
                        f"""
                        <div class="prediction-box">
                            <h2 style="margin:0; color: #1E88E5;">🤟 {ctx.video_processor.current_prediction}</h2>
                            <p style="font-size: 1.5rem; margin:10px 0 0 0;" class="{conf_class}">
                                Confidence: {ctx.video_processor.current_confidence*100:.1f}%
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    # Show top 3 predictions
                    if ctx.video_processor.top3_predictions:
                        with top3_placeholder.container():
                            st.markdown("**Top 3 Predictions:**")
                            for i, (label, conf) in enumerate(
                                ctx.video_processor.top3_predictions, 1
                            ):
                                st.write(f"{i}. **{label}**: {conf*100:.1f}%")
                else:
                    prediction_placeholder.info("👋 Show a hand sign to start!")


if __name__ == "__main__":
    main()
