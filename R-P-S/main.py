import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

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

            overlay = frame.copy()
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 0), -1)
            alpha = 0.3
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            cv2.rectangle(frame,(xmin,ymin),(xmax, ymax),(255,0,0),2)
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
