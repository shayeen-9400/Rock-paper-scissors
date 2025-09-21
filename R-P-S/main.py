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

# Load trained model
model = load_model(r'C:\Users\shahe\Documents\GitHub\Rock-paper-scissors\model\rps_model.h5')
labels = ['paper', 'rock', 'scissors']

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1620)   # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280) 

def determine_winner(move1, move2):
    if move1 == move2:
        return "Draw"
    elif (move1 == 'rock' and move2 == 'scissors') or \
         (move1 == 'scissors' and move2 == 'paper') or \
         (move1 == 'paper' and move2 == 'rock'):
        return "Player 1"
    else:
        return "Player 2"

last_move1, last_move2 = "", ""
last_winner = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    h, w, _ = frame.shape
    hand_predictions = []

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
        hand_positions = []

        for hand_landmarks in result.multi_hand_landmarks:
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(x_list) * w), int(max(x_list) * w)
            ymin, ymax = int(min(y_list) * h), int(max(y_list) * h)

            offset = 20
            xmin = max(0, xmin - offset)
            ymin = max(0, ymin - offset)
            xmax = min(w, xmax + offset)
            ymax = min(h, ymax + offset)

            hand_img = frame[ymin:ymax, xmin:xmax]

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

            try:
                resized = cv2.resize(hand_img, (64, 64))
                normalized = resized / 255.0
                input_tensor = np.expand_dims(normalized, axis=0)

                prediction = model.predict(input_tensor, verbose=0)
                class_id = np.argmax(prediction)
                class_label = labels[class_id]

                hand_positions.append((xmin, class_label, (xmin, ymin)))

                cv2.putText(frame, class_label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print("Prediction error:", e)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Sort hands left to right by xmin
        hand_positions.sort(key=lambda x: x[0])
        if len(hand_positions) == 2:
            (x1, move1, pos1), (x2, move2, pos2) = hand_positions
            hand_predictions = [(move1, pos1), (move2, pos2)]

            # Player labels
            cv2.putText(frame, "Player 1", (pos1[0], pos1[1] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Player 2", (pos2[0], pos2[1] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Winner update
            if move1 != last_move1 or move2 != last_move2:
                last_winner = determine_winner(move1, move2)
                last_move1, last_move2 = move1, move2

    # Heading
    cv2.putText(frame, "Rock Paper Scissors Game", (80, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (25, 255, 55), 3)

    # Display winner at bottom center
    if last_winner:
        text = f"{last_winner} wins!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 30
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 0, 255), 3)

    cv2.imshow("Rock Paper Scissors - Two Players", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
