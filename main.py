import face
import asyncio
import cv2

camera=cv2.VideoCapture(0)
mesh=face.face_mesh(0, cv2camerainput=camera)

if __name__ == "__main__":
  asyncio.run(mesh.meshify())