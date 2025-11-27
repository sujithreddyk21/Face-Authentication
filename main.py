
# main.py
import os
import io
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import uvicorn

from model_utils import FaceModels
from utils import select_largest_face, cosine_similarity


# -----------------------------
# Configuration
# -----------------------------
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
MAX_IMAGE_SIZE_BYTES = int(os.getenv("MAX_IMAGE_SIZE_BYTES", 5 * 1024 * 1024))


# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(title="Face Authentication Service")


# Global model instance
models = None


# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
def startup_event():
    global models
    device = os.getenv("DEVICE", "cpu")
    models = FaceModels(device=device)
    print("🚀 Models loaded successfully on device:", device)


# -----------------------------
# Response Model
# -----------------------------
class VerifyResponse(BaseModel):
    verification: str
    similarity: float
    threshold: float
    metric: str
    faces: dict
    used_faces: dict
    embedding_dim: int


# -----------------------------
# Helper: Read uploaded image
# -----------------------------
def read_imagefile_to_numpy(file: UploadFile) -> np.ndarray:
    contents = file.file.read()

    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(413, detail="File too large")

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        return np.array(img)
    except Exception:
        raise HTTPException(400, detail="Invalid image file")


# -----------------------------
# /verify Endpoint
# -----------------------------
@app.post("/verify", response_model=VerifyResponse)
async def verify(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    threshold: Optional[float] = Query(None),
    face_index_a: Optional[int] = Query(0),
    face_index_b: Optional[int] = Query(0),
    metric: Optional[str] = Query("cosine")
):
    # Load images
    img_a = read_imagefile_to_numpy(image_a)
    img_b = read_imagefile_to_numpy(image_b)

    # Detect faces
    faces_a = models.detect(img_a)
    faces_b = models.detect(img_b)

    if len(faces_a) == 0:
        return JSONResponse(status_code=422, content={"error": "No faces detected in image_a", "code": "NO_FACE_DETECTED"})

    if len(faces_b) == 0:
        return JSONResponse(status_code=422, content={"error": "No faces detected in image_b", "code": "NO_FACE_DETECTED"})

    # Choose face based on index or largest face
    def choose_face(faces, idx):
        if idx is not None and 0 <= idx < len(faces):
            return faces[idx], idx
        return select_largest_face(faces)

    selected_face_a, idx_a = choose_face(faces_a, face_index_a)
    selected_face_b, idx_b = choose_face(faces_b, face_index_b)

    # Crop and align (resize)
    crop_a = models.crop_align(img_a, selected_face_a, size=(160, 160))
    crop_b = models.crop_align(img_b, selected_face_b, size=(160, 160))

    # Extract embeddings
    emb_a = models.get_embedding(crop_a)
    emb_b = models.get_embedding(crop_b)

    # Compute similarity
    sim = cosine_similarity(emb_a, emb_b)
    used_threshold = threshold if threshold is not None else THRESHOLD

    # Decide
    verification = "same person" if sim >= used_threshold else "different person"

    # Final response
    return {
        "verification": verification,
        "similarity": sim,
        "threshold": used_threshold,
        "metric": metric,
        "faces": {"image_a": faces_a, "image_b": faces_b},
        "used_faces": {"image_a": idx_a, "image_b": idx_b},
        "embedding_dim": models.embedding_dim
    }


# -----------------------------
# Run with: python main.py
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
