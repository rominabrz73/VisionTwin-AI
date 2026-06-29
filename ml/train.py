"""Transfer-learning entry point for crack/no-crack classification."""

import argparse
import json
import random
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.evaluate import evaluate_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VisionTwin AI with ResNet18.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = [transforms.Resize((224, 224)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    datasets_by_split = {
        "train": datasets.ImageFolder(args.data_dir / "train", transforms.Compose([
            transforms.RandomHorizontalFlip(), *common
        ])),
        "val": datasets.ImageFolder(args.data_dir / "val", transforms.Compose(common)),
    }
    if datasets_by_split["train"].classes != ["crack", "no_crack"]:
        raise ValueError("Expected class folders named 'crack' and 'no_crack'.")
    loaders = {name: DataLoader(ds, args.batch_size, shuffle=name == "train", num_workers=0)
               for name, ds in datasets_by_split.items()}

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    mlflow.set_experiment("vision-twin-ai")
    with mlflow.start_run():
        mlflow.log_params(vars(args) | {"device": str(device), "architecture": "resnet18"})
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for images, labels in loaders["train"]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            loss = running_loss / len(datasets_by_split["train"])
            mlflow.log_metric("train_loss", loss, step=epoch)
            print(f"Epoch {epoch + 1}/{args.epochs} - loss: {loss:.4f}")

        ROOT.joinpath("models").mkdir(exist_ok=True)
        checkpoint = ROOT / "models" / "resnet18_crack_classifier.pt"
        torch.save({"model_state_dict": model.state_dict(), "classes": datasets_by_split["train"].classes}, checkpoint)
        metrics = evaluate_model(model, loaders["val"], device, ROOT / "reports")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(checkpoint), artifact_path="models")
        mlflow.log_artifacts(str(ROOT / "reports"), artifact_path="reports")
        mlflow.pytorch.log_model(model, "pytorch-model")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
