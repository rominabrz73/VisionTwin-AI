"""Inference service with lightweight visual evidence generation."""

import base64
import sys
from pathlib import Path

import cv2
import torch
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.preprocess import decode_image, preprocess_image  # noqa: E402

MODEL_PATH = ROOT / "models" / "resnet18_crack_classifier.pt"


class Predictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        if model_path.exists():
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            self.model = model.to(self.device).eval()

    def predict(self, image_bytes: bytes) -> dict:
        bgr = decode_image(image_bytes)
        rgb, summary = preprocess_image(bgr)
        evidence_image, processed_image, crack_area, evidence_score = self._build_evidence(bgr)
        summary["detected_crack_area_percent"] = crack_area
        summary["visual_evidence_score"] = evidence_score
        if self.model is not None:
            with torch.no_grad():
                probabilities = torch.softmax(self.model(self.transform(rgb).unsqueeze(0).to(self.device)), dim=1)[0]
            index = int(probabilities.argmax())
            return {"prediction": ["crack", "no crack"][index], "confidence": round(float(probabilities[index]), 4),
                    "model_source": "resnet18", "preprocessing": summary,
                    "processed_image": processed_image,
                    "evidence_image": self._render_evidence(evidence_image, index == 0)}

        # Allows the UI to be explored before training; never presented as model inference.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edge_ratio = summary["edge_pixels"] / (224 * 224)
        contrast = float(gray.std()) / 128
        score = min(0.95, max(0.05, 0.25 + edge_ratio * 2.5 + contrast * 0.2))
        prediction = "crack" if score >= 0.5 else "no crack"
        return {"prediction": prediction, "confidence": round(score if prediction == "crack" else 1 - score, 4),
                "model_source": "demo_heuristic", "preprocessing": summary,
                "processed_image": processed_image,
                "evidence_image": self._render_evidence(evidence_image, prediction == "crack")}

    @staticmethod
    def _build_evidence(image):
        resized = cv2.resize(image, (640, 420), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        canvas = resized.copy()
        relevant = [contour for contour in contours if 12 <= cv2.arcLength(contour, False) <= 1400]
        contour_area = sum(cv2.contourArea(contour) for contour in relevant)
        area_percent = min(100.0, contour_area / (640 * 420) * 100)
        edge_ratio = float(cv2.countNonZero(edges)) / (640 * 420)
        evidence_score = min(100.0, edge_ratio * 650 + min(len(relevant), 100) * 0.25)
        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(processed, (0, 0), (640, 38), (20, 29, 27), -1)
        cv2.putText(processed, "OPENCV CANNY EDGE PROCESSING", (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)
        success, encoded = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not success:
            raise ValueError("Could not generate processed evidence.")
        processed_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
        return (canvas, edges, relevant), processed_url, round(area_percent, 2), round(evidence_score, 1)

    @staticmethod
    def _render_evidence(evidence, crack_detected: bool) -> str:
        canvas, edges, contours = evidence
        if crack_detected:
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                if max(width, height) >= 18:
                    cv2.rectangle(canvas, (x, y), (x + width, y + height), (42, 55, 232), 2)
            label, colour = "AI REVIEW REGION", (42, 55, 232)
        else:
            edge_colour = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edge_colour[:, :, 0] = 0
            edge_colour[:, :, 2] = 0
            canvas = cv2.addWeighted(canvas, 0.78, edge_colour, 0.7, 0)
            label, colour = "EDGE MAP - NO CRACK CLASSIFICATION", (75, 145, 56)
        cv2.rectangle(canvas, (0, 0), (640, 38), (20, 29, 27), -1)
        cv2.putText(canvas, label, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        success, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not success:
            raise ValueError("Could not generate visual evidence.")
        return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
