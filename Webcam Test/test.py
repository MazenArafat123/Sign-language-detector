# ============================================================
# Real-time ASL Recognition with Word Building
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model
import time

# --- Load Model and Classes ---
model = load_model(r"E:\Computer Vision\DEPI_Hand Gesture Recognition\Sign-language-detector\Models\asl_landmarks_final.h5")
with open(r"E:\Computer Vision\DEPI_Hand Gesture Recognition\Sign-language-detector\Models\asl_landmarks_classes.pkl", 'rb') as f:
    class_names = pickle.load(f)

print(f"Model loaded. Classes: {class_names}")

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
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Webcam started. Press 'q' to quit.")
print("Hold a letter for 2 seconds to add it to the word")
print("Show 'DEL' gesture to delete last character")
print("Press 'c' to clear current word")
print("Press 's' to save word to history")

# Word building variables
current_word = ""
word_history = []
prediction_history = []
history_size = 5

# Letter confirmation variables
last_stable_letter = None
stable_letter_start_time = None
confirmation_threshold = 2.0  # seconds to hold a letter to confirm

# Cooldown to prevent duplicate letters
last_added_time = 0
cooldown_duration = 0.5  # seconds

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
    
    current_time = time.time()
    predicted_class = None
    confidence = 0
    
    if results.multi_hand_landmarks:
        # HAND DETECTED -> Predict gesture
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
            final_prediction = max(set(prediction_history), key=prediction_history.count)
            predicted_class = class_names[final_prediction]
            
            # Letter confirmation logic
            if confidence > 0.7:  # Only consider high confidence predictions
                if predicted_class == last_stable_letter:
                    # Same letter is being held
                    time_held = current_time - stable_letter_start_time
                    
                    # Check if letter should be added
                    if time_held >= confirmation_threshold and (current_time - last_added_time) > cooldown_duration:
                        if predicted_class.upper() == "SPACE":
                            current_word += " "
                            last_added_time = current_time
                            stable_letter_start_time = current_time  # Reset timer
                        elif predicted_class.upper() == "DEL":
                            if current_word:
                                current_word = current_word[:-1]
                                last_added_time = current_time
                                stable_letter_start_time = current_time  # Reset timer
                        elif predicted_class.upper() not in ["NOTHING", "DELETE", "BACKSPACE", "SPACE", "DEL"]:
                            # Add letter (convert to lowercase for words)
                            current_word += predicted_class.lower()
                            last_added_time = current_time
                            stable_letter_start_time = current_time  # Reset timer
                else:
                    # New letter detected
                    last_stable_letter = predicted_class
                    stable_letter_start_time = current_time
            
            # Display current prediction with hold indicator
            text = f"{predicted_class}: {confidence*100:.1f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.rectangle(frame, (10, 10), (20 + text_size[0], 60), (0, 0, 0), -1)
            
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.putText(frame, text, (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Show hold progress bar if same letter
            if last_stable_letter == predicted_class and confidence > 0.7:
                time_held = current_time - stable_letter_start_time
                progress = min(time_held / confirmation_threshold, 1.0)
                bar_width = 300
                bar_height = 20
                bar_x = 15
                bar_y = 70
                
                # Background bar
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                # Progress bar
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
                cv2.putText(frame, "Hold to add letter", (bar_x, bar_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # NO HAND DETECTED
        cv2.putText(frame, "NOTHING", (15, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        prediction_history.clear()
        last_stable_letter = None
    
    # Display current word being built
    word_display = current_word if current_word else "[empty]"
    cv2.rectangle(frame, (10, h - 150), (w - 10, h - 70), (0, 0, 0), -1)
    cv2.putText(frame, "Current Word:", (20, h - 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, word_display, (20, h - 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # Display word history
    if word_history:
        history_text = " | ".join(word_history[-3:])  # Show last 3 words
        cv2.putText(frame, f"History: {history_text}", (20, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # Instructions
    instructions = [
        "Press 'q' to quit | 'c' to clear | 's' to save | DEL gesture to delete",
        "Hold letter for 2 sec to add"
    ]
    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction, (20, h - 30 + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('ASL Word Builder', frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        current_word = ""
        print("Word cleared")
    elif key == ord('s'):
        if current_word.strip():
            word_history.append(current_word.strip())
            print(f"Saved word: '{current_word.strip()}'")
            current_word = ""
        else:
            print("No word to save")

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()

# Print final word history
if word_history:
    print("\nWord History:")
    for i, word in enumerate(word_history, 1):
        print(f"  {i}. {word}")
else:
    print("\nNo words were saved")

print("Webcam closed.")