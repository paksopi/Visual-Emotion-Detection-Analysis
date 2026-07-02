"""Track A runner: DeepFace, DeepFace.analyze(actions=['emotion']).

FER2013 images are pre-cropped 48x48 faces, so we use detector_backend='skip'
+ enforce_detection=False (bypass DeepFace's own detector/aligner) for the
same reason documented in run_fer.py: there's no scene for a detector to
search, and detection overhead isn't measurable on this dataset.
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval import LatencyTimer, VRAMTracker, classification_metrics, RunLogger  # noqa: E402

LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABEL_MAP = {
    "angry": "angry",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprised": "surprise",
    "neutral": "neutral",
}
MANIFEST = REPO_ROOT / "data" / "fer2013" / "test_manifest_shuffled.csv"


def main(limit: int | None = None):
    from deepface import DeepFace

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    if limit:
        rows = rows[:limit]

    logger = RunLogger(model_name="deepface", track="A")
    timer = LatencyTimer()
    y_true, y_pred = [], []

    with VRAMTracker(use_torch=False) as vram:
        for i, row in enumerate(rows):
            img_path = REPO_ROOT / row["path"]
            true_label = LABEL_MAP[row["label"]]
            img = cv2.imread(str(img_path))
            with timer.measure():
                result = DeepFace.analyze(
                    img,
                    actions=["emotion"],
                    detector_backend="skip",
                    enforce_detection=False,
                    silent=True,
                )
            pred_label = result[0]["dominant_emotion"]
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
    }
    out = logger.write_summary(summary)
    logger.close()
    print("accuracy:", metrics["accuracy"], "macro_f1:", metrics["macro_f1"])
    print("latency:", timer.summary())
    print("summary written to", out)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)
