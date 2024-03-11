import face
import pose as bodypos
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
    pose=bodypos.PoseEstimator(0, cv2camerainput=camera)

    while camera.isOpened():
        ret, frame = camera.read()

        if not ret:
            print("Error: Could not capture frame.")
            break

        frame_copy = frame.copy()

        mesh_frame = await mesh.meshify(frame_copy)
        pose_frame = await pose.fetch_pose(mesh_frame)
        hand_frame = await hand.gesturify(pose_frame)
        image = flip(hand_frame, 1)
        imshow('Combined', image)
        

        if waitKey(1) & 0xFF == ord('q'):
            break

    await hand.camera_release()
    await mesh.camera_release()
    await pose.camera_release()

if __name__ == "__main__":
  asyncio.run(main())