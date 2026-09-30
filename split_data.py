import argparse
import random
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description="Split the Concrete Crack Images dataset into train/val folders.")
parser.add_argument("--src", required=True, help="Folder that contains the Positive and Negative folders")
parser.add_argument("--dst", default="data", help="Output folder (default: data)")
args = parser.parse_args()

SRC = Path(args.src)
DST = Path(args.dst)
random.seed(42)

for name in ("Positive", "Negative"):
    if not (SRC / name).is_dir():
        raise SystemExit(f"Folder not found: {SRC / name}")

for src_name, cls in [("Positive", "crack"), ("Negative", "no_crack")]:
    files = sorted((SRC / src_name).glob("*.jpg"))
    random.shuffle(files)
    cut = int(len(files) * 0.8)  # 80% train, 20% validation
    for split, subset in [("train", files[:cut]), ("val", files[cut:])]:
        out = DST / split / cls
        out.mkdir(parents=True, exist_ok=True)
        for f in subset:
            shutil.copy(f, out / f.name)
        print(f"{split}/{cls}: {len(subset)} images")

print("done")