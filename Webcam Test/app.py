import os
import warnings
import logging
import sys

# Suppress ALL warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model
import time
from PIL import Image

# ============ CONFIGURATION - UPDATE THESE PATHS ============
MODEL_PATH = r"E:\Computer Vision\DEPI_Hand Gesture Recognition\Sign-language-detector\Models\asl_landmarks_final.h5"
CLASSES_PATH = r"E:\Computer Vision\DEPI_Hand Gesture Recognition\Sign-language-detector\Models\asl_landmarks_classes.pkl"
# ============================================================

# Page configuration
st.set_page_config(
    page_title="ASL Word Builder",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .word-display {
        font-size: 2.5rem;
        color: #00FF00;
        background-color: #000000;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-family: monospace;
    }
    .prediction-box {
        font-size: 1.8rem;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .instruction-box {
        background-color: black;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'word_history' not in st.session_state:
    st.session_state.word_history = []
if 'last_stable_letter' not in st.session_state:
    st.session_state.last_stable_letter = None
if 'stable_letter_start_time' not in st.session_state:
    st.session_state.stable_letter_start_time = None
if 'last_added_time' not in st.session_state:
    st.session_state.last_added_time = 0
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

@st.cache_resource
def load_asl_model():
    """Load the trained model and class names from fixed paths"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found at: {MODEL_PATH}")
            return None, None
        
        if not os.path.exists(CLASSES_PATH):
            st.error(f"❌ Classes file not found at: {CLASSES_PATH}")
            return None, None
        
        model = load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'rb') as f:
            class_names = pickle.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

@st.cache_resource
def initialize_mediapipe():
    """Initialize MediaPipe Hands"""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return mp_hands, mp_drawing, hands

def extract_landmarks_from_frame(hand_landmarks):
    """Extract 63 features from detected hand"""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords)

def process_frame(frame, model, class_names, hands, mp_hands, mp_drawing, 
                 confirmation_threshold=2.0, cooldown_duration=0.5, history_size=5):
    """Process a single frame and return prediction"""
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    current_time = time.time()
    predicted_class = None
    confidence = 0
    time_held = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            # Extract landmarks for prediction
            landmarks = extract_landmarks_from_frame(hand_landmarks)
            landmarks = landmarks.reshape(1, -1)
            
            # Predict
            predictions = model.predict(landmarks, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            
            # Smooth predictions
            st.session_state.prediction_history.append(predicted_idx)
            if len(st.session_state.prediction_history) > history_size:
                st.session_state.prediction_history.pop(0)
            
            # Use most common prediction
            final_prediction = max(set(st.session_state.prediction_history), 
                                 key=st.session_state.prediction_history.count)
            predicted_class = class_names[final_prediction]
            
            # Letter confirmation logic
            if confidence > 0.7:
                if predicted_class == st.session_state.last_stable_letter:
                    time_held = current_time - st.session_state.stable_letter_start_time
                    
                    if time_held >= confirmation_threshold and \
                       (current_time - st.session_state.last_added_time) > cooldown_duration:
                        
                        if predicted_class.upper() == "SPACE":
                            st.session_state.current_word += " "
                            st.session_state.last_added_time = current_time
                            st.session_state.stable_letter_start_time = current_time
                        elif predicted_class.upper() == "DEL":
                            if st.session_state.current_word:
                                st.session_state.current_word = st.session_state.current_word[:-1]
                                st.session_state.last_added_time = current_time
                                st.session_state.stable_letter_start_time = current_time
                        elif predicted_class.upper() not in ["NOTHING", "DELETE", "BACKSPACE"]:
                            st.session_state.current_word += predicted_class.lower()
                            st.session_state.last_added_time = current_time
                            st.session_state.stable_letter_start_time = current_time
                else:
                    st.session_state.last_stable_letter = predicted_class
                    st.session_state.stable_letter_start_time = current_time
            
            if st.session_state.last_stable_letter == predicted_class and confidence > 0.7:
                time_held = current_time - st.session_state.stable_letter_start_time
    else:
        predicted_class = "NOTHING"
        st.session_state.prediction_history.clear()
        st.session_state.last_stable_letter = None
    
    return frame, predicted_class, confidence, time_held

# Main content
st.markdown('<h1 class="main-header">🤟 ASL Word Builder</h1>', unsafe_allow_html=True)

# Load model automatically
with st.spinner("Loading ASL model..."):
    model, class_names = load_asl_model()

if model is not None and class_names is not None:

    # Initialize MediaPipe
    mp_hands, mp_drawing, hands = initialize_mediapipe()
    
    # Sidebar for settings
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/sign-language.png", width=100)
        st.title("⚙️ Settings")
        
        st.info(f"📁 **Model:** Loaded\n\n📊 **Classes:** {len(class_names)}")
        
        st.divider()
        
        # Parameters
        st.subheader("🎛️ Parameters")
        confirmation_threshold = st.slider("Hold Time (seconds)", 0.5, 5.0, 2.0, 0.5)
        cooldown_duration = st.slider("Cooldown (seconds)", 0.1, 2.0, 0.5, 0.1)
        
        st.divider()
        
        # Camera selection
        camera_index = st.number_input("Camera Index", 0, 10, 0)       
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed")
        video_placeholder = st.empty()
        
    with col2:
        st.subheader("📊 Prediction")
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        st.subheader("📝 Current Word")
        word_placeholder = st.empty()
        
        # Action buttons
        st.subheader("🎮 Actions")
        if st.button("💾 Save", use_container_width=True):
            if st.session_state.current_word.strip():
                st.session_state.word_history.append(st.session_state.current_word.strip())
                st.success(f"Saved: '{st.session_state.current_word.strip()}'")
                st.session_state.current_word = ""
                st.rerun()
  
        st.subheader("📜 Word History")
        history_placeholder = st.empty()
    
    # Instructions
    with st.expander("📖 Instructions", expanded=True):
        st.markdown("""
        <div class="instruction-box">
        <h4>How to Use:</h4>
        <ul>
            <li>🤚 Hold your hand in front of the camera</li>
            <li>✋ Make an ASL letter sign</li>
            <li>⏱️ Hold the sign for 2 seconds to add it to the word</li>
            <li>🔤 Special gestures: SPACE (add space), DEL (delete last character)</li>
            <li>💾 Click "Save" to add the word to history</li>
            <li>🗑️ Click "Clear" to start a new word</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Camera controls
    col_start, col_stop = st.columns(2)
    with col_start:
        start_button = st.button("▶️ Start Camera", use_container_width=True, type="primary")
    with col_stop:
        stop_button = st.button("⏹️ Stop Camera", use_container_width=True)
    
    if start_button:
        st.session_state.camera_running = True
    if stop_button:
        st.session_state.camera_running = False
    
    # Run camera
    if st.session_state.camera_running:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, predicted_class, confidence, time_held = process_frame(
                frame, model, class_names, hands, mp_hands, mp_drawing,
                confirmation_threshold, cooldown_duration
            )
            
            # Display video
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Display prediction
            if predicted_class:
                color = "green" if confidence > 0.7 else "orange"
                prediction_placeholder.markdown(
                    f'<div class="prediction-box" style="background-color: {color}; color: white;">'
                    f'{predicted_class}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                confidence_placeholder.metric("Confidence", f"{confidence*100:.1f}%")
                
                # Progress bar
                if time_held > 0:
                    progress = min(time_held / confirmation_threshold, 1.0)
                    progress_placeholder.progress(progress, text=f"Hold: {time_held:.1f}s / {confirmation_threshold}s")
                else:
                    progress_placeholder.empty()
            
            # Display current word
            word_display = st.session_state.current_word if st.session_state.current_word else "[empty]"
            word_placeholder.markdown(
                f'<div class="word-display">{word_display}</div>',
                unsafe_allow_html=True
            )
            
            # Display historys
            if st.session_state.word_history:
                history_text = "\n".join([f"{i+1}. {word}" for i, word in enumerate(st.session_state.word_history[-5:])])
                history_placeholder.markdown(f"**Last 5 Words:**\n\n```\n{history_text}\n```")
            
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()
        st.info("Camera stopped")
    
else:
    st.error("❌ Failed to load model. Please check the file paths at the top of the script.")
    st.info("**Current paths:**")
    st.code(f"Model: {MODEL_PATH}\nClasses: {CLASSES_PATH}", language="text")
    st.warning("⚠️ Update the MODEL_PATH and CLASSES_PATH variables at the top of the script with your correct file paths.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>ASL Word Builder | Built with Streamlit 🎈 | Powered by MediaPipe & TensorFlow</p>
</div>
""", unsafe_allow_html=True)