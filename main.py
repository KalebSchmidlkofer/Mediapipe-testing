import face
import gestures
import asyncio
from cv2 import VideoCapture
#? The Plan
#* The right solution is to merge multiple graphs into one graph and keep everything in one frame processor/calculator graph.
#* The quick solution is to run two frame processors/calculator graph instances for face and hand tracking separately in one app.


camera=VideoCapture(0)
mesh=face.face_mesh(0, cv2camerainput=camera)
hand=gestures.gestures(0, cv2camerainput=camera)

async def main():
  await asyncio.gather(mesh.meshify(), hand.gesturify())

if __name__ == "__main__":
  asyncio.run(main())