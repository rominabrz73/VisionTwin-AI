from typing import Literal

from pydantic import BaseModel


class PreprocessingSummary(BaseModel):
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    edge_pixels: int
    contour_count: int
    threshold_mean: float
    detected_crack_area_percent: float
    visual_evidence_score: float


class PredictionResponse(BaseModel):
    prediction: Literal["crack", "no crack"]
    confidence: float
    model_source: Literal["resnet18", "demo_heuristic"]
    preprocessing: PreprocessingSummary
    processed_image: str
    evidence_image: str
