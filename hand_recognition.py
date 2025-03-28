import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands() 
mp_draw = mp.solutions.drawing_utils  


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    result = hands.process(rgb_frame)

 
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            #Convert to pixel coordinates
            h, w, _ = frame.shape
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
            ring_y = int(ring_tip.y * h)
            pinky_y = int(pinky_tip.y * h)
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

            # Detect Thumbs-Up Gesture
            if (thumb_y < index_y and thumb_y < middle_y and
                ring_y > thumb_y and pinky_y > thumb_y ):
                cv2.putText(frame, "Thumbs Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
             
            # Detect Thumbs-Down Gesture 
            if (thumb_y > index_y and thumb_y > middle_y and
                ring_y < thumb_y and pinky_y < thumb_y and thumb_y > wrist_y):
                cv2.putText(frame, "Thumbs Down!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            #Detect Peace Sign
            if (index_y < middle_y and  
                ring_y > middle_y and pinky_y > middle_y and  # Ring & Pinky Bent
                abs(index_x - middle_x) > 30):  # Ensure fingers are apart
                cv2.putText(frame, "Peace!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Calculate distance between thumb tip and index tip
            distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            # Detect OK Sign (👌)
            if (distance < 30 and middle_y < wrist_y and 
                ring_y < wrist_y and pinky_y < wrist_y and thumb_y<wrist_y):
                cv2.putText(frame, "Okkie!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
