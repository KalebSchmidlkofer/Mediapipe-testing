import face
import asyncio

mesh=face.face_mesh(0)

if __name__ == "__main__":
  asyncio.run(mesh.meshify())