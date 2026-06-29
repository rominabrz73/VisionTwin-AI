"""Shared OpenCV preprocessing for training, inspection, and inference."""

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreprocessingSummary:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    edge_pixels: int
    contour_count: int
    threshold_mean: float


def decode_image(image_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a valid image.")
    return image


def preprocess_image(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> tuple[np.ndarray, dict]:
    """Return a model-ready RGB image and an explainable CV summary."""
    original_height, original_width = image.shape[:2]
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    summary = PreprocessingSummary(
        original_width=original_width,
        original_height=original_height,
        resized_width=size[0],
        resized_height=size[1],
        edge_pixels=int(np.count_nonzero(edges)),
        contour_count=len(contours),
        threshold_mean=round(float(threshold.mean()), 2),
    )
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), asdict(summary)


def load_and_preprocess(path: str | Path) -> tuple[np.ndarray, dict]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return preprocess_image(image)
