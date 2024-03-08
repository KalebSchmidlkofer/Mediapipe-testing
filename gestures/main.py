import mediapipe as mp
import cv2
import os
import asyncio
import time

# mpsol=mp.solutions.mediapipe.python.solutions #? MediaPipeSolutions
# Path to the gesture recognizer model
model_path = os.path.join(os.getcwd(), 'gesture_recognizer.task')

# Create a GestureRecognizer instance with the live stream mode
options = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp.solutions.hands.Hands()
# hands.Hands
start_time = time.time()
frame_count = 0


class camera():
  async def __init__(self, camera):
    self.cap = cv2.VideoCapture(camera)  # Use 0 for the default camera  

  async def gestures(self):
    while self.cap.isOpened():
        success, image = self.cap.read()
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

        cv2.imshow('MediaPipe Hands', image)


async def face_mesh():
  model_path = os.path.join(os.getcwd(), 'face_landmarker.task')
  mp_face_landmark = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
  
  cap = cv2.VideoCapture(0)
    
  if not cap.isOpened():
      print("Error: Could not open camera.")
      exit()
  
  while True:
      ret, frame = cap.read()
  
      if not ret:
          print("Error: Could not capture frame.")
          break
        
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
      results = mp_face_landmark.process(rgb_frame)
  
      if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
              for landmark in face_landmarks.landmark:
                  x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                  cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
  
      # Display the frame
      cv2.imshow('Face Landmarks', frame)
  
      # Check for key press
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
        
  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  # asyncio.run(gestures())
  asyncio.run(face_mesh())
