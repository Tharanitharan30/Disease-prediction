from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def detect_organ_from_filename(filename: str) -> str:
    lower = filename.lower()
    if "brain" in lower or "mri" in lower or "tumor" in lower:
        return "Brain"
    if "liver" in lower:
        return "Liver"
    if "kidney" in lower:
        return "Kidney"
    if "lung" in lower:
        return "Lung"
    if "heart" in lower:
        return "Heart"
    return "Brain"


def load_rgb_image(image_bytes: bytes, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def image_statistics_features(image_arr: np.ndarray) -> np.ndarray:
    # Fixed-length fallback feature vector so tabular models can be called safely if compatible.
    gray = image_arr.mean(axis=2)
    features = np.array(
        [
            image_arr.mean(),
            image_arr.std(),
            image_arr[:, :, 0].mean(),
            image_arr[:, :, 1].mean(),
            image_arr[:, :, 2].mean(),
            gray.mean(),
            gray.std(),
            np.percentile(gray, 25),
            np.percentile(gray, 50),
            np.percentile(gray, 75),
        ],
        dtype=np.float32,
    )
    return features.reshape(1, -1)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[4]
