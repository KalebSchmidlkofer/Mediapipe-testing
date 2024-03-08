import cv2
import mediapipe as mp
import numpy as np
import os
from sys import platform
model_path = os.path.join(os.getcwd(), 'face_landmarker.task')

# Initialize MediaPipe Face Landmarker
mp_face_landmark = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Initialize OpenCV VideoCapture with camera index 1

# if platform=='win32':
  # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# else:
  # cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(0)
  
# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read frame from the camera
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Error: Could not capture frame.")
        break

    # Convert the frame to RGB as MediaPipe works with RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Landmarker
    results = mp_face_landmark.process(rgb_frame)

    # If face landmarks are detected, draw them on the frame
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

# Release the VideoCapture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()