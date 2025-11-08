# ============================================================
# Real-time ASL Recognition with Landmarks (Webcam)
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model

# --- Load Model and Classes ---
model = load_model("C:\\Users\\Nour\\Downloads\\asl_landmarks_final.h5")
with open("C:\\Users\\Nour\\Downloads\\asl_landmarks_classes.pkl", 'rb') as f:
    class_names = pickle.load(f)

print(f"✅ Model loaded. Classes: {class_names}")

# --- Initialize MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Extract Landmarks Function ---
def extract_landmarks_from_frame(image_rgb, hand_landmarks):
    """Extract 63 features from detected hand"""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords)

# --- Start Webcam ---
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🎥 Webcam started. Press 'q' to quit.")

# For smoothing predictions
prediction_history = []
history_size = 5

# --- Modified Webcam Section ---

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # HAND DETECTED -> Predict gesture from trained 28 classes
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
            landmarks = extract_landmarks_from_frame(image_rgb, hand_landmarks)
            landmarks = landmarks.reshape(1, -1)  # Shape: (1, 63)
            
            # Predict
            predictions = model.predict(landmarks, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            
            # Smooth predictions
            prediction_history.append(predicted_idx)
            if len(prediction_history) > history_size:
                prediction_history.pop(0)
            
            # Use most common prediction
            final_prediction = max(set(prediction_history), key=prediction_history.count)
            predicted_class = class_names[final_prediction]
            
            # Display prediction on frame
            text = f"{predicted_class}: {confidence*100:.1f}%"
            
            # Background rectangle for text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.rectangle(frame, (10, 10), (20 + text_size[0], 60), (0, 0, 0), -1)
            
            # Text color based on confidence
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.putText(frame, text, (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Show top 3 predictions
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            y_offset = 100
            for i, idx in enumerate(top3_indices):
                label = f"{class_names[idx]}: {predictions[0][idx]*100:.1f}%"
                cv2.putText(frame, label, (15, y_offset + i*35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    else:
        # NO HAND DETECTED -> Show "NOTHING"
        cv2.putText(frame, "NOTHING", (15, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, "No hand detected", (15, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        prediction_history.clear()  # Clear history when no hand
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit", (w - 250, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('ASL Recognition (Landmarks)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
print("✅ Webcam closed.")