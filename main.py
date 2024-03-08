import face
import gestures
import asyncio
from cv2 import VideoCapture, imshow, waitKey, flip
#? The Plan
#* The right solution is to merge multiple graphs into one graph and keep everything in one frame processor/calculator graph.
#* The quick solution is to run two frame processors/calculator graph instances for face and hand tracking separately in one app.


async def main():
    camera = VideoCapture(0)
    mesh=face.face_mesh(0, cv2camerainput=camera)
    hand=gestures.gestures(0, cv2camerainput=camera)

    while camera.isOpened():
        ret, frame = camera.read()

        if not ret:
            print("Error: Could not capture frame.")
            break

        frame_copy = frame.copy()

        mesh_frame = await mesh.meshify(frame_copy)
        hand_meshed_frame = await hand.gesturify(mesh_frame)

        image = flip(hand_meshed_frame, 1)
        imshow('Combined', image)
        

        if waitKey(1) & 0xFF == ord('q'):
            break

    await hand.camera_release()
    await mesh.camera_release()

if __name__ == "__main__":
  asyncio.run(main())