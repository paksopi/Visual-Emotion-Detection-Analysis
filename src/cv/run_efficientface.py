"""Track A runner: EfficientFace (zengqunzhao/EfficientFace), RAF-DB checkpoint.

Research code, no pip package: model class + checkpoint pulled manually per
plan §1 caveat (data/models/efficientface_repo is a shallow clone of the
upstream repo for models/EfficientFace.py + models/modulator.py; checkpoint
from the author's Google Drive link in the README, via gdown).

Label-order caveat: the upstream repo trains via torchvision.datasets.ImageFolder
on RAF-DB, and neither main.py nor test&vis/test.py documents the resulting
class index order. The commonly-cited RAF-DB numeric convention (Surprise,
Fear, Disgust, Happy, Sad, Angry, Neutral) was tried first and does NOT match
this checkpoint's actual output order (verified empirically — see below), so
whatever folder-naming scheme the author actually used produces a different
permutation. The real mapping was inferred empirically: ran raw (unmapped)
predictions against FER2013 ground truth for the first 1000 shuffled images,
built a true-label x raw-index confusion matrix, and solved the optimal
raw-index -> label assignment via the Hungarian algorithm (maximizing diagonal
agreement). Result: 0=neutral, 1=happy, 2=sad, 3=surprise, 4=fear, 5=disgust,
6=angry, with 51.6% agreement on the calibration set — consistent with this
model being a real (if imperfect, cross-domain) classifier rather than the
assignment fitting noise. To avoid calibrating and reporting on the same data,
final metrics below are computed on the held-out remainder of the shuffled
manifest (rows 1000+), not the calibration slice.
"""
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
EFFICIENTFACE_REPO = REPO_ROOT / "data" / "models" / "efficientface_repo"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EFFICIENTFACE_REPO))

from eval import LatencyTimer, VRAMTracker, classification_metrics, RunLogger  # noqa: E402

LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
DATASET_LABEL_MAP = {
    "angry": "angry",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprised": "surprise",
    "neutral": "neutral",
}
# empirically inferred via Hungarian-assignment calibration, see module docstring
RAFDB_IDX_TO_LABEL = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]

CHECKPOINT = REPO_ROOT / "data" / "models" / "efficientface" / "rafdb.pth"
MANIFEST = REPO_ROOT / "data" / "fer2013" / "test_manifest_shuffled.csv"


def load_model(device):
    from models.EfficientFace import efficient_face

    model = efficient_face()
    model.fc = nn.Linear(1024, 7)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = {k.replace("module.", "", 1): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main(limit: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    normalize = transforms.Normalize(
        mean=[0.57535914, 0.44928582, 0.40079932],
        std=[0.20735591, 0.18981615, 0.18132027],
    )
    preprocess = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    rows = rows[1000:]  # rows[:1000] used for label-order calibration, held out here
    if limit:
        rows = rows[:limit]

    logger = RunLogger(model_name="efficientface", track="A")
    timer = LatencyTimer()
    y_true, y_pred = [], []

    with VRAMTracker(use_torch=True) as vram, torch.no_grad():
        for i, row in enumerate(rows):
            img_path = REPO_ROOT / row["path"]
            true_label = DATASET_LABEL_MAP[row["label"]]
            img = Image.open(img_path).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            with timer.measure():
                logits = model(x)
                pred_idx = int(logits.argmax(dim=1).item())
            pred_label = RAFDB_IDX_TO_LABEL[pred_idx]
            y_true.append(true_label)
            y_pred.append(pred_label)
            logger.record(image=str(img_path), true=true_label, pred=pred_label)
            if i % 500 == 0:
                print(f"{i}/{len(rows)}")

    metrics = classification_metrics(y_true, y_pred, labels=LABELS)
    summary = {
        "metrics": metrics,
        "latency": timer.summary(),
        "peak_vram_mb": vram.peak_mb(),
        "n_images": len(rows),
        "note": "RAF-DB label-index order assumed, not confirmed against upstream training code",
    }
    out = logger.write_summary(summary)
    logger.close()
    print("accuracy:", metrics["accuracy"], "macro_f1:", metrics["macro_f1"])
    print("latency:", timer.summary())
    print("summary written to", out)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)
