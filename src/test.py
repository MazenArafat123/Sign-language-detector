# ========================================
# REAL-TIME ASL RECOGNITION WITH CAMERA + MEDIAPIPE
# ========================================

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import mediapipe as mp
import time

# ========================================
# CONFIGURATION
# ========================================

MODEL_PATH = r"models\asl_gesture_model_final.h5"

CLASS_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CONFIDENCE_THRESHOLD = 0.6
PREDICTION_BUFFER_SIZE = 5

COLOR_CONFIDENT = (0, 255, 0)
COLOR_UNCERTAIN = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)

# ========================================
# LOAD MODEL
# ========================================

print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ========================================
# MEDIAPIPE HAND DETECTOR
# ========================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

mp_draw = mp.solutions.drawing_utils

# ========================================
# HELPER FUNCTIONS
# ========================================


def preprocess_frame(frame):
    """Resize + normalize the image like training"""
    img = cv2.resize(frame, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def get_smoothed_prediction(predictions, prediction_buffer):
    """Smooth predictions across frames"""
    prediction_buffer.append(predictions[0])
    avg_predictions = np.mean(prediction_buffer, axis=0)
    predicted_class = np.argmax(avg_predictions)
    confidence = avg_predictions[predicted_class]
    return predicted_class, confidence


def draw_text_with_background(frame, text, position, font_scale=1, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(frame,
                  (x - 5, y - text_height - 5),
                  (x + text_width + 5, y + baseline + 5),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, COLOR_TEXT, thickness)


# ========================================
# MAIN CAMERA LOOP
# ========================================

def run_camera_recognition():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Could not open camera")
        return

    prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    print("\n==============================")
    print("ASL RECOGNITION STARTED ")
    print("==============================")
    print("📹 Camera active")
    print("🖐️  Show your hand clearly")
    print("⌨️  Press 'q' to quit")
    print("==============================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        roi = None
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get bounding box around the hand
                h, w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min = int(min(x_coords) * w) - 30
                x_max = int(max(x_coords) * w) + 30
                y_min = int(min(y_coords) * h) - 30
                y_max = int(max(y_coords) * h) + 30

                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_max, w)
                y_max = min(y_max, h)

                roi = frame[y_min:y_max, x_min:x_max]
                cv2.rectangle(frame, (x_min, y_min),
                              (x_max, y_max), COLOR_CONFIDENT, 2)

        if roi is not None and roi.size > 0:
            processed_roi = preprocess_frame(roi)
            predictions = model.predict(processed_roi, verbose=0)
            predicted_class, confidence = get_smoothed_prediction(
                predictions, prediction_buffer)
            predicted_label = CLASS_LABELS[predicted_class]
            is_confident = confidence > CONFIDENCE_THRESHOLD
            color = COLOR_CONFIDENT if is_confident else COLOR_UNCERTAIN

            if is_confident:
                main_text = f"{predicted_label}"
                conf_text = f"{confidence:.1%}"
            else:
                main_text = "Uncertain"
                conf_text = f"Max: {confidence:.1%}"

            draw_text_with_background(
                frame, main_text, (50, 50), font_scale=2, thickness=3)
            draw_text_with_background(
                frame, conf_text, (50, 90), font_scale=1, thickness=2)

        else:
            draw_text_with_background(
                frame, "No hand detected", (50, 50), font_scale=1, thickness=2)

        # FPS
        fps_counter += 1
        if fps_counter >= 10:
            fps = 10 / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_counter = 0

        h, w, _ = frame.shape
        draw_text_with_background(
            frame, f"FPS: {fps:.1f}", (w - 150, 40), font_scale=0.7, thickness=2)

        cv2.imshow("ASL Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed successfully")


if __name__ == "__main__":
    run_camera_recognition()
