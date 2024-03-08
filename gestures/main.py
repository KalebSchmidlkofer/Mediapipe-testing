import mediapipe as mp
import cv2
import os
import asyncio
from typing import Optional

model_path = os.path.join(os.getcwd(), 'gesture_recognizer.task')

class gestures:
  def __init__(self, camera_input_index, max_hands=2, detection_confidance=0.5, tracking_confidance=0.5, cv2camerainput: Optional[cv2.VideoCapture] = None):
    self.options = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=max_hands, min_detection_confidence=tracking_confidance, min_tracking_confidence=detection_confidance)
    self.hands = mp.solutions.hands.Hands()
    
    if cv2camerainput is not None:
      self.cap = cv2camerainput
    else:
      self.cap = cv2.VideoCapture(camera_input_index)



  async def _cap_check(self):
    if not self.cap.isOpened():
      print("Error: Could not open camera.")
      exit()

  async def gesturify(self):
    while self.cap.isOpened():
      success, image = self.cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue

      image = cv2.flip(image, 1)

      rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      results = self.hands.process(rgb_image)

      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

      cv2.imshow('MediaPipe Hands', image)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        await self.camera_release()

  async def camera_release(self):
    self.cap.release()
    cv2.destroyAllWindows()
          


if __name__ == "__main__":
  hand=gestures(camera_input_index=0)
  asyncio.run(hand.gesturify())