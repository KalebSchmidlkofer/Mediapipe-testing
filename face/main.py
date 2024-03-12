import cv2
import mediapipe as mp
import numpy as np
import os
from sys import platform
from typing import Optional
import asyncio
import threading

model_path = os.path.join(os.getcwd(), '*.task')
#! mpsol=mp.solutions.mediapipe.python.solutions #? MediaPipeSolutions

class face_mesh:
  def __init__(self, camera_input_index: int, max_num_faces: Optional[int] = 1, cv2camerainput: Optional[cv2.VideoCapture] = None):
    self.cap = cv2camerainput
    self.mp_face_landmark = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=max_num_faces)
    

  async def _cap_check(self):
    if not self.cap.isOpened():
      print("Error: Could not open camera.")
      exit()
  
  async def start_threading(self):
    self.thread=threading.Thread()

  async def meshify(self, frame):
    ret, frame = self.cap.read()

    if not ret:
      print("Error: Could not capture frame.")
      return frame

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = self.mp_face_landmark.process(rgb_frame)
    results = await asyncio.get_event_loop().run_in_executor(None, self.mp_face_landmark.process, rgb_frame)

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
          x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
          cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('Face_mesh', cv2.flip(frame, 1))
  
  async def camera_release(self):
    if not self.cap == None:
      self.cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
  # mesh=face_mesh(0)
  # asyncio.run(mesh.meshify())
  pass