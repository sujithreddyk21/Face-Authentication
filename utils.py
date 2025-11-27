import numpy as np

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def select_largest_face(faces):
    if not faces:
        return None, None
    areas = [box_area(f["box"]) for f in faces]
    idx = int(np.argmax(areas))
    return faces[idx], idx

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
# utils.py

import numpy as np

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def select_largest_face(faces):
    if not faces:
        return None, None
    areas = [box_area(f["box"]) for f in faces]
    idx = int(np.argmax(areas))
    return faces[idx], idx

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
