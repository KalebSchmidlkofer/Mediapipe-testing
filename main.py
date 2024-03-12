import face
import pose as bodypos
import gestures
import asyncio
from cv2 import VideoCapture, imshow, waitKey, flip
import threading
#? The Plan
#* The right solution is to merge multiple graphs into one graph and keep everything in one frame processor/calculator graph.
#* The quick solution is to run two frame processors/calculator graph instances for face and hand tracking separately in one app.


async def main():
    camera = VideoCapture(0)
    mesh=face.face_mesh(0, cv2camerainput=camera)
    hand=gestures.gestures(0, cv2camerainput=camera)
    pose=bodypos.PoseEstimator(0, cv2camerainput=camera)

    while camera.isOpened():
        ret, frame = camera.read()

        if not ret:
            print("Error: Could not capture frame.")
            break

        frame_copy = frame.copy()

        async_tasks=[
        # mesh.meshify(frame_copy),
        pose.fetch_pose(frame_copy),
        hand.gesturify(frame_copy)
        ]
        await asyncio.gather(*async_tasks)

        if waitKey(1) & 0xFF == ord('q'):
            break

    await hand.camera_release()
    await mesh.camera_release()
    await pose.camera_release()

if __name__ == "__main__":
  asyncio.run(main())