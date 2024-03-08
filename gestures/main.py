import mediapipe as mp
import cv2
import os
import asyncio


# Path to the gesture recognizer model
model_path = os.path.join(os.getcwd(), 'gesture_recognizer.task')

# Create a GestureRecognizer instance with the live stream mode
options = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp.solutions.hands.Hands()

# OpenCV setup
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

async def video():
  while cap.isOpened():
      success, image = cap.read()
      if not success:
          print("Ignoring empty camera frame.")
          continue

      # Flip the image horizontally for a later selfie-view display
      image = cv2.flip(image, 1)

      # Convert the BGR image to RGB
      rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Process the image with MediaPipe Hands
      results = hands.process(rgb_image)

      if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
              # Draw hand landmarks on the image
              mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

      # Display the resulting image
      cv2.imshow('MediaPipe Hands', image)

      # Break the loop when 'q' is pressed
      if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
  asyncio.run(video())