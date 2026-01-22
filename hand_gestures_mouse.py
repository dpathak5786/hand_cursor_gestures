import cv2
import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

prev_x, prev_y = 0, 0
first_frame = True
click_cooldown = 0  # Prevent spam clicking

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    h, w, _ = frame.shape
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
            
            # CURSOR MOVEMENT (Index finger tip = landmark 8)
            if len(lm_list) >= 9:
                index_x, index_y = lm_list[8][1], lm_list[8][2]
                
                raw_screen_x = np.interp(index_x, (50, w-50), (0, screen_width))
                raw_screen_y = np.interp(index_y, (50, h-50), (0, screen_height))
                
                if first_frame:
                    smooth_x, smooth_y = raw_screen_x, raw_screen_y
                    first_frame = False
                else:
                    dx = abs(raw_screen_x - prev_x)
                    dy = abs(raw_screen_y - prev_y)
                    if dx > 8 or dy > 8:
                        smooth_x = prev_x + 0.3 * (raw_screen_x - prev_x)
                        smooth_y = prev_y + 0.3 * (raw_screen_y - prev_y)
                    else:
                        smooth_x, smooth_y = prev_x, prev_y
                
                pyautogui.moveTo(int(smooth_x), int(smooth_y))
                prev_x, prev_y = smooth_x, smooth_y
                
                cv2.circle(frame, (index_x, index_y), 12, (0, 255, 0), -1)
            
            # PINCH DETECTION - FIXED VERSION
            if len(lm_list) >= 5:  # Need thumb tip (4) and index tip (8)
                thumb_tip = lm_list[4]      # Thumb tip
                index_tip = lm_list[8]      # Index finger tip
                
                # Calculate EXACT distance between thumb and index tips
                distance = np.sqrt((thumb_tip[1] - index_tip[1])**2 + 
                                 (thumb_tip[2] - index_tip[2])**2)
                
                # SHOW DISTANCE ON SCREEN (for debugging)
                cv2.putText(frame, f"Pinch: {int(distance)}", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # CLICK when distance is small (thumb touches index)
                if distance < 30 and click_cooldown <= 0:
                    pyautogui.click()
                    click_cooldown = 10  # 0.3 sec cooldown
                    cv2.circle(frame, (thumb_tip[1], thumb_tip[2]), 15, (0, 0, 255), -1)
                    cv2.putText(frame, "CLICKED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                elif click_cooldown > 0:
                    click_cooldown -= 1
    
    # Instructions on screen
    cv2.putText(frame, "click thumb+index se hota hai", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Q se exit hota hai", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("mast hai na ye small project", frame)
    
    if cv2.waitKey(1)==113:
        break

cap.release()
cv2.destroyAllWindows()
