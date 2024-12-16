import cv2
import mediapipe as mp
import numpy as np
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
def draw_overlay(frame, hand_landmarks, width, height):
    overlay = frame.copy()
    thickness = 2
    color = (255, 255, 255)
    points = []
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * width), int(landmark.y * height)
        points.append((x, y))
    if len(points) >= 8:
        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        cv2.line(overlay, thumb_tip, index_tip, color, thickness)
        cv2.line(overlay, index_tip, middle_tip, color, thickness)
        cv2.line(overlay, middle_tip, ring_tip, color, thickness)
        cv2.circle(overlay, thumb_tip, 10, color, thickness)
        cv2.circle(overlay, index_tip, 10, color, thickness)
        cv2.circle(overlay, middle_tip, 10, color, thickness)
        cv2.circle(overlay, ring_tip, 10, color, thickness)
    return overlay
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                frame = draw_overlay(frame, hand_landmarks, width, height)
        cv2.imshow("Real-Time Hand Tracking with Overlay", frame)
        if cv2.waitKey(1) & 0xFF == ord('v'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
