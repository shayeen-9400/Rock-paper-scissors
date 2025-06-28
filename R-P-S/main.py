import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'best_model_v2.h5')
model = load_model(model_path)

class_names = ['paper', 'rock', 'scissors']
def resize_with_padding(img, size=(64, 64)):
    old_size = img.shape[:2]  # (height, width)
    ratio = float(size[0]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    resized = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = size[1] - new_size[1]
    delta_h = size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]  # Black padding
    new_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1620)   # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280) 

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame")
        continue
    
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for better visualization
    h,w,_ = frame.shape
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            x_cords =[landmark.x * w for landmark in hand_landmarks.landmark]
            y_cords = [landmark.y * h for landmark in hand_landmarks.landmark]

            xmin,xmax =int(min(x_cords)), int(max(x_cords))
            ymin, ymax = int(min(y_cords)), int(max(y_cords))

            xmin, xmax, ymin, ymax = max(0, xmin), min(w, xmax), max(0, ymin), min(h,ymax)

            hand_img = frame[ymin:ymax, xmin:xmax]

            if hand_img.shape[0]>0 and hand_img.shape[1]>0:
                hand_resized = resize_with_padding(hand_img, (64, 64))
                hand_normalized = hand_resized / 255.0
                hand_input = np.expand_dims(hand_normalized, axis=0)

                prediction = model.predict(hand_input)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction)


                overlay = frame.copy()
                cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 0), -1)
                alpha = 0.3
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                cv2.rectangle(frame,(xmin,ymin),(xmax, ymax),(255,0,0),2)

                label = f"{predicted_class} ({confidence*100:.1f}%)"
                label_y = max(0,ymin-10)
                cv2.putText(frame, label, (xmin, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)


    cv2.imshow("RPS Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
