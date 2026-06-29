# VisionTwin AI

An AI-assisted infrastructure inspection and Digital Twin-style **decision-support prototype**. Upload a surface image, run explainable OpenCV processing and ResNet18 classification, associate the result with an asset component, and turn it into a risk level, health score, recommended action, inspection history, and downloadable engineering summary.

> **Scope:** this is a Digital Twin-style concept demonstration—not a full BIM, LiDAR, structural assessment, or safety certification system.

## Why this fits applied AI and Digital Twin roles

VisionTwin AI demonstrates the useful seam between ML research and engineering delivery: Computer Vision, OpenCV, ResNet18 transfer learning, Model Validation, Digital Twin Concepts, AI Dashboard Integration, Decision Support, and an MLOps-ready workflow. The component map shows how model output can become asset context and an inspection priority rather than remain an isolated prediction.

The prototype supports inspection prioritisation: it helps a maintenance team decide what to review first. It does **not** replace civil or structural engineers, site investigation, or a certified structural assessment.

## Decision-support workflow

- Classifies each image as crack or no crack and records model confidence.
- Converts inference into Low, Medium, High, or Critical risk with a recommended response.
- Maintains a session-local React inspection history and component-level condition state for Roof, Wall A, Wall B, and Floor.
- Calculates building health, open issues, and high-risk areas from the latest component inspections.
- Returns an OpenCV evidence overlay, candidate crack area, and visual evidence score for interpretability.
- Generates a downloadable HTML inspection report with an explicit engineer-review disclaimer.

## Architecture

```text
Public concrete crack images
          │
          ▼
 OpenCV preprocessing ──► ResNet18 training ──► validation reports
 resize · gray · blur          │                 accuracy · P/R/F1
 Canny · threshold · contour   ├──► MLflow       confusion matrix
                               ▼
                       model checkpoint
                               │
User image ──► React ──► FastAPI /predict ──► prediction + confidence
                 │                                  │
                 └──── Digital Twin-style asset view ◄──┘
                       Roof · Wall A · Wall B · Floor
```

## Quick start with Docker

Requirements: Docker Desktop and Docker Compose.

```bash
git clone <your-repository-url>
cd vision-twin-ai
docker compose up --build
```

Open `http://localhost:5173`; API docs are at `http://localhost:8000/docs`. If no checkpoint exists, the API returns a clearly identified `demo_heuristic` result so the product flow remains demonstrable. Train the model for actual ResNet18 inference.

## Local development

Use Python 3.11. From the project root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r ml/requirements.txt -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

For local quality checks, install `backend/requirements-dev.txt`, then run `ruff check backend ml` and `pytest backend/tests`.

## Dataset and directory layout

Use the public **Concrete Crack Images for Classification** dataset (40,000 images, balanced positive/negative examples), available through Kaggle. Accept its license/terms, download it locally, and split images into:

```text
data/
├── train/
│   ├── crack/
│   └── no_crack/
└── val/
    ├── crack/
    └── no_crack/
```

Keep `data/` out of Git. A simple 80/20 stratified split is sufficient for this demonstration. The folder names are intentional: `ImageFolder` maps `crack` to class 0 and `no_crack` to class 1.

## Train and track an experiment

```bash
python ml/train.py --data-dir data --epochs 3 --batch-size 16
mlflow ui
```

Training freezes the pretrained ResNet18 feature extractor and learns a two-output classification head. It writes `models/resnet18_crack_classifier.pt`, logs parameters, loss, final metrics, the PyTorch model, and report artifacts to MLflow. Visit `http://localhost:5000` for the local MLflow UI.

## Model validation metrics

The validation step calculates accuracy, precision, recall, and F1-score. It also creates:

- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`

No headline metrics are claimed in this repository because results depend on the exact split and training run. After training, record the MLflow run ID and replace this paragraph with your measured holdout results. For safety-oriented inspection, discuss recall for the `crack` class alongside false negatives—not accuracy alone.

## OpenCV pipeline

`ml/preprocess.py` is shared by training support and inference. It resizes to 224×224, converts to grayscale, applies Gaussian blur, extracts Canny edges, computes Otsu thresholding, and detects external contours. The API returns contour count, edge-pixel count, threshold mean, and dimensions as a concise, explainable preprocessing summary. ResNet18 receives the resized RGB image; the handcrafted stages provide inspection evidence rather than replacing learned features.

## MLOps choices

- Pinned Python dependencies and deterministic random seeds improve repeatability.
- MLflow records parameters, training loss, validation metrics, checkpoint, and reports.
- Model and report directories make artifact ownership explicit.
- Docker Compose provides consistent API/dashboard startup.
- GitHub Actions runs Ruff, API tests, and the production frontend build.
- A model-source field prevents the UI fallback from being confused with trained inference.
- The dashboard exposes model version, local inference type, validation coverage, and honest `MLflow-ready` lifecycle metadata rather than claiming a production registry.

## API example

```bash
curl -X POST http://localhost:8000/predict -F "file=@inspection.jpg"
```

```json
{
  "prediction": "crack",
  "confidence": 0.91,
  "model_source": "resnet18",
  "preprocessing": {
    "original_width": 640, "original_height": 480,
    "resized_width": 224, "resized_height": 224,
    "edge_pixels": 3812, "contour_count": 47, "threshold_mean": 133.4,
    "detected_crack_area_percent": 1.84,
    "visual_evidence_score": 73.2
  },
  "processed_image": "data:image/jpeg;base64,<encoded edge image>",
  "evidence_image": "data:image/jpeg;base64,<encoded overlay>"
}
```

## Limitations and future work

- A public close-up dataset does not represent every material, camera, weather, or lighting condition; validate on site-specific data.
- A random image split may leak near-duplicate texture patterns. Group by structure/site where metadata permits.
- Classification localises neither crack geometry nor severity. Add segmentation or detection with calibrated uncertainty.
- This is a Digital Twin-style component view, not a full BIM or LiDAR Digital Twin.
- Asset selection is manual; the model does not infer whether an image belongs to the roof, floor, or a particular wall.
- Inspection history is React session state only; it resets on refresh and is not shared between users or devices.
- The contour overlay is supporting visual evidence, not pixel-accurate crack segmentation.
- An industrial version could integrate drone imagery, IFC/BIM assets, LiDAR, maintenance history, cloud storage, and persistent asset records.
- Production work needs drift monitoring, model/version promotion, security controls, accessibility testing, and human review thresholds.
- Connect results to IFC/BIM or geospatial asset records; add LiDAR only where its 3D evidence solves a defined need.

## Suggested CV bullets

- Built an end-to-end AI-assisted infrastructure inspection prototype using OpenCV, PyTorch ResNet18, FastAPI, and React/TypeScript.
- Implemented model validation with precision, recall, F1, confusion matrices, MLflow experiment tracking, and reproducible model artifacts.
- Translated computer-vision predictions into Digital Twin-style component health, risk tiers, recommended engineering actions, and inspection reports.
- Delivered an interpretable decision-support dashboard with OpenCV evidence overlays, local inspection history, Docker Compose, and GitHub Actions CI.

## Repository map

```text
backend/    FastAPI service, typed schemas, inference, tests
ml/         OpenCV preprocessing, training, evaluation
frontend/   React + TypeScript asset dashboard
models/     generated model checkpoints (Git-ignored)
reports/    generated validation artifacts (Git-ignored)
```

## Responsible use

This prototype supports screening and portfolio discussion only. A prediction must not replace inspection by a qualified engineer.
