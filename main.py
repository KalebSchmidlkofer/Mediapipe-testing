import face
import gestures
import asyncio
from cv2 import VideoCapture

camera=VideoCapture(0)
mesh=face.face_mesh(0, cv2camerainput=camera)
hand=gestures.gestures(0, cv2camerainput=camera)

if __name__ == "__main__":
  asyncio.run(mesh.meshify())
  asyncio.run(hand.gesturify())