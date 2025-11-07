import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model  # type: ignore
import time
from PIL import Image

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
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
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


# --- Initialize MediaPipe ---
@st.cache_resource
def init_mediapipe():
    """Initialize MediaPipe Hands"""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return mp_hands, mp_drawing, hands


# --- Extract Landmarks ---
def extract_landmarks_from_frame(hand_landmarks):
    """Extract 63 features from detected hand"""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords)


# --- Process Frame ---
def process_frame(
    frame,
    hands,
    model,
    class_names,
    mp_hands,
    mp_drawing,
    prediction_history,
    history_size=5,
):
    """Process a single frame and return predictions"""
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    predicted_class = "NOTHING"
    confidence = 0.0
    top3_predictions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
            )

            # Extract landmarks for prediction
            landmarks = extract_landmarks_from_frame(hand_landmarks)
            landmarks = landmarks.reshape(1, -1)

            # Predict
            predictions = model.predict(landmarks, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]

            # Smooth predictions
            prediction_history.append(predicted_idx)
            if len(prediction_history) > history_size:
                prediction_history.pop(0)

            # Use most common prediction
            final_prediction = max(
                set(prediction_history), key=prediction_history.count
            )
            predicted_class = class_names[final_prediction]

            # Get top 3 predictions
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_predictions = [
                (class_names[idx], predictions[0][idx]) for idx in top3_indices
            ]

            # Display prediction on frame
            text = f"{predicted_class}: {confidence*100:.1f}%"
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.rectangle(frame, (10, 10),
                          (20 + text_size[0], 60), (0, 0, 0), -1)

            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.putText(frame, text, (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    else:
        # No hand detected
        cv2.putText(
            frame, "NOTHING", (15,
                               50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3
        )
        cv2.putText(
            frame,
            "No hand detected",
            (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        prediction_history.clear()

    return frame, predicted_class, confidence, top3_predictions


# --- Main App ---
def main():
    # Header
    st.markdown(
        '<p class="main-header">🤟 ASL Recognition System</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Real-time American Sign Language Recognition using AI</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model paths
        st.subheader("📁 Model Configuration")
        model_path = st.text_input(
            "Model Path",
            value=r"models/asl_landmarks_final.h5",
            help="Path to the trained .h5 model file",
        )

        classes_path = st.text_input(
            "Classes Path",
            value=r"models/asl_landmarks_classes.pkl",
            help="Path to the classes pickle file",
        )

        # Camera settings
        st.subheader("📹 Camera Settings")
        camera_index = st.number_input(
            "Camera Index", min_value=0, max_value=5, value=0
        )

        # Detection settings
        st.subheader("🎯 Detection Settings")
        min_detection_confidence = st.slider(
            "Min Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
        )

        history_size = st.slider(
            "Prediction Smoothing",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of frames to smooth predictions",
        )

        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown(
            """
        1. Load the model using the button below
        2. Click **Start Camera** to begin
        3. Show ASL hand signs to the camera
        4. View real-time predictions
        5. Click **Stop Camera** when done
        """
        )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Camera Feed")
        video_placeholder = st.empty()
        status_placeholder = st.empty()

    with col2:
        st.subheader("📊 Predictions")
        prediction_placeholder = st.empty()
        top3_placeholder = st.empty()
        stats_placeholder = st.empty()

    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        load_model_btn = st.button("🔄 Load Model", use_container_width=True)

    with col_btn2:
        start_btn = st.button("▶️ Start Camera", use_container_width=True)

    with col_btn3:
        stop_btn = st.button("⏹️ Stop Camera", use_container_width=True)

    # Initialize session state
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # Load model
    if load_model_btn:
        with st.spinner("Loading model..."):
            model, class_names = load_asl_model(model_path, classes_path)
            if model is not None:
                st.session_state.model = model
                st.session_state.class_names = class_names
                st.session_state.model_loaded = True
                st.success(
                    f"✅ Model loaded successfully! Classes: {len(class_names)}")
            else:
                st.error("❌ Failed to load model. Check the file paths.")

    # Start camera
    if start_btn:
        if not st.session_state.model_loaded:
            st.error("⚠️ Please load the model first!")
        else:
            st.session_state.camera_running = True

    # Stop camera
    if stop_btn:
        st.session_state.camera_running = False
        st.session_state.prediction_history = []
        status_placeholder.info("📷 Camera stopped")

    # Camera loop
    if st.session_state.camera_running and st.session_state.model_loaded:
        # Initialize MediaPipe
        mp_hands, mp_drawing, hands = init_mediapipe()

        # Open camera
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        status_placeholder.success("🎥 Camera is running...")

        frame_count = 0
        start_time = time.time()

        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("❌ Failed to grab frame")
                break

            # Process frame
            processed_frame, predicted_class, confidence, top3_predictions = (
                process_frame(
                    frame,
                    hands,
                    st.session_state.model,
                    st.session_state.class_names,
                    mp_hands,
                    mp_drawing,
                    st.session_state.prediction_history,
                    history_size,
                )
            )

            # Convert BGR to RGB for display
            processed_frame_rgb = cv2.cvtColor(
                processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(
                processed_frame_rgb, channels="RGB", use_column_width=True
            )

            # Update predictions
            with prediction_placeholder.container():
                if predicted_class != "NOTHING":
                    conf_class = (
                        "confidence-high" if confidence > 0.7 else "confidence-low"
                    )
                    st.markdown(
                        f"""
                    <div class="prediction-box">
                        <h2 style="margin:0; color: #1E88E5;">🤟 {predicted_class}</h2>
                        <p style="font-size: 1.5rem; margin:10px 0 0 0;" class="{conf_class}">
                            Confidence: {confidence*100:.1f}%
                        </p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("👋 No hand detected - Show a hand sign!")

            # Show top 3 predictions
            if top3_predictions:
                with top3_placeholder.container():
                    st.markdown("**Top 3 Predictions:**")
                    for i, (label, conf) in enumerate(top3_predictions, 1):
                        st.write(f"{i}. **{label}**: {conf*100:.1f}%")

            # Stats
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            with stats_placeholder.container():
                st.markdown("---")
                col_stat1, col_stat2 = st.columns(2)
                col_stat1.metric("FPS", f"{fps:.1f}")
                col_stat2.metric("Frames", frame_count)

            # Small delay
            time.sleep(0.03)

            # Check if stop button was pressed
            if not st.session_state.camera_running:
                break

        # Release camera
        cap.release()
        hands.close()
        status_placeholder.info("📷 Camera stopped")


if __name__ == "__main__":
    main()
