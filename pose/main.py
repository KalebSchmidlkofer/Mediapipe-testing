import cv2
import mediapipe as mp
import numpy as np
import os
from typing import Optional
import asyncio

model_path = os.path.join(os.getcwd(), '*.task')

class PoseEstimator:
    def __init__(self, camera_input_index: int, model_complexity: Optional[int] = 1, cv2camerainput: Optional[cv2.VideoCapture] = None):
        self.failed_frames = 0
        if cv2camerainput is not None:
            self.cap = cv2camerainput
        else:
            self.cap = cv2.VideoCapture(camera_input_index)
        self.mp_pose_landmark = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=model_complexity)
    
    async def _cap_check(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            exit()

    async def fetch_pose(self, frame):
      
        success, image = self.cap.read()
        if not success:
            print("Error: Could not capture frame.")
            return frame

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = await asyncio.get_event_loop().run_in_executor(None, self.mp_pose_landmark.process, rgb_frame)

        if results is not None:
          if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
              frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        return frame

    async def camera_release(self):
        self.cap.release()
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     pose_estimator = PoseEstimator(camera_input_index=0)
#     loop = asyncio.get_event_loop()
#     try:
#         loop.run_until_complete(pose_estimator._cap_check())
#         while True:
#             frame, results = loop.run_until_complete(pose_estimator.fetch_pose())
#             if frame is None:
#                 break

#             if results is not None:
#                 if results.pose_landmarks:
#                     mp.solutions.drawing_utils.draw_landmarks(
#                         frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                
#                 image = cv2.flip(frame, 1)
#                 cv2.imshow("Pose Estimation", image)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         loop.run_until_complete(pose_estimator.camera_release())
