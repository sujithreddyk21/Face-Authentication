# model_utils.py

import os
import numpy as np
import cv2
import torch
from typing import Dict, Any, Optional
from facenet_pytorch import MTCNN, InceptionResnetV1

DEVICE = os.getenv("DEVICE", "cpu")


class FaceModels:
    """
    Loads:
    - MTCNN face detector
    - InceptionResnetV1 embedder (FaceNet)
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or DEVICE
        self._init_detector()
        self._init_embedder()

    # -------------------------
    # Load MTCNN detector
    # -------------------------
    def _init_detector(self):
        self.detector = MTCNN(keep_all=True, device=self.device)

    # -------------------------
    # Load FaceNet embedder
    # -------------------------
    def _init_embedder(self):
        self.embedder = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to(self.device)
        )
        self.embedding_dim = 512

    # -------------------------
    # Face Detection
    # -------------------------
    def detect(self, rgb_image: np.ndarray):
        from PIL import Image

        img = Image.fromarray(rgb_image)

        boxes, probs, landmarks = self.detector.detect(img, landmarks=True)
        faces = []

        if boxes is None:
            return faces

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            score = float(probs[i]) if probs is not None else None
            lm = (
                [[float(px), float(py)] for px, py in landmarks[i]]
                if landmarks is not None
                else None
            )

            faces.append({"box": [x1, y1, x2, y2], "score": score, "landmarks": lm})

        return faces

    # -------------------------
    # Crop & Resize
    # -------------------------
    def crop_align(self, rgb_image: np.ndarray, face: Dict[str, Any], size=(160, 160)):
        x1, y1, x2, y2 = face["box"]
        h, w = rgb_image.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = rgb_image[y1:y2, x1:x2]

        if crop.size == 0:
            raise ValueError("Invalid face crop region")

        face_img = cv2.resize(crop, size)
        return face_img

    # -------------------------
    # Embeddings
    # -------------------------
    def get_embedding(self, face_rgb: np.ndarray) -> np.ndarray:
        img = face_rgb.astype(np.float32)

        tensor = (
            torch.tensor(img)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        tensor = (tensor - 127.5) / 128.0

        with torch.no_grad():
            emb = self.embedder(tensor)

        emb = emb.squeeze(0).cpu().numpy().astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb
