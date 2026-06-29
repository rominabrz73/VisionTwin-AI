"""Model validation and report generation utilities."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, precision_recall_fscore_support)


def evaluate_model(model, data_loader, device, report_dir: Path) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            output = model(images.to(device))
            predictions.extend(output.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="binary", pos_label=0, zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(targets, predictions, target_names=["crack", "no_crack"], zero_division=0)
    (report_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    ConfusionMatrixDisplay.from_predictions(targets, predictions, display_labels=["crack", "no crack"])
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    return metrics
