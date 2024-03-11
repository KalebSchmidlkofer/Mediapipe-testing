import mediapipe as mp
import cv2
import os
import asyncio
from typing import Optional

model_path = os.path.join(os.getcwd(), '*.task')

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

  async def gesturify(self, frame):
    success, image = self.cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      return frame

    # image = cv2.flip(image, 1)

    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # results = self.hands.process(rgb_image)
    results = await asyncio.get_event_loop().run_in_executor(None, self.hands.process, rgb_frame)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    return frame

  async def camera_release(self):
    self.cap.release()
    cv2.destroyAllWindows()
          


if __name__ == "__main__":
  # hand=gestures(camera_input_index=0)
  # asyncio.run(hand.gesturify())
  pass