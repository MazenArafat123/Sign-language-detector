import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model(r'models\asl_gesture_model_MobileNetV2.h5')


class_labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}

cap = cv2.VideoCapture(0)
word = ""   

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)


    img = cv2.resize(roi, (128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    
    pred = model.predict(img)
    predicted_class = int(np.argmax(pred))
    predicted_label = class_labels.get(predicted_class, "?")  

   
    cv2.putText(frame, f"Letter: {predicted_label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    key = cv2.waitKey(1)
    if key == ord('a'):  
        word += predicted_label
    elif key == ord('c'):  
        word = ""
    elif key == ord('q'):  
        break

    cv2.putText(frame, f"Word: {word}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("ASL Recognition", frame)

cap.release()
cv2.destroyAllWindows()
