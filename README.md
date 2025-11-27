# Face Authentication Service 🔐

A production-ready FastAPI microservice for face verification using FaceNet embeddings and cosine similarity. It takes two face images, detects faces with MTCNN, extracts 512‑D embeddings using InceptionResnetV1, and returns a similarity score along with a verification decision.

---

## Features

- 🔍 Face detection with MTCNN (multi-face support, returns bounding boxes and landmarks). [attached_file:3]
- 🧠 512‑dimensional FaceNet embeddings using `InceptionResnetV1` pretrained on VGGFace2. [attached_file:3][web:9]
- 📐 Cosine similarity–based verification with configurable threshold (via query param or `THRESHOLD` env var). [attached_file:1][attached_file:2][web:19]
- 🖼 Multiple faces per image with automatic largest-face selection or explicit face index. [attached_file:1][attached_file:2]
- 🧾 Clean JSON response including
