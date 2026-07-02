"""Track A runner: `fer` library (justinshenk/fer), which ships oarriaga's
FER2013 mini-xception weights under a TFLite-quantized wrapper with its own
Haar-cascade face detector.

Note on FER2013 images: the dataset ships pre-cropped 48x48 grayscale faces,
so there is no "scene" for a face detector to search — the library's own
Haar cascade fails to find a face in an image that IS already just a face at
that resolution (min_face_size default is 50px > the 48px image). We bypass
detection by passing the full-image bounding box explicitly, which still
exercises the library's real preprocessing (resize/offset/normalize) and
classifier path — just skips the detection step, whose overhead can't be
measured on this dataset (would need uncropped scene frames; see plan §2's
self-collected stress set, out of scope this run).
"""
import csv
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval import LatencyTimer, VRAMTracker, classification_metrics, RunLogger  # noqa: E402

LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
# dataset manifest label -> fer library label
LABEL_MAP = {
    "angry": "angry",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprised": "surprise",
    "neutral": "neutral",
}
CASCADE_PATH = REPO_ROOT / "data" / "models" / "haarcascade_frontalface_default.xml"
MANIFEST = REPO_ROOT / "data" / "fer2013" / "test_manifest_shuffled.csv"


def main(limit: int | None = None):
    from fer.fer import FER

    detector = FER(mtcnn=False, cascade_file=str(CASCADE_PATH))

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    if limit:
        rows = rows[:limit]

    logger = RunLogger(model_name="fer_lib", track="A")
    timer = LatencyTimer()
    y_true, y_pred = [], []

    with VRAMTracker(use_torch=False) as vram:
        for i, row in enumerate(rows):
            img_path = REPO_ROOT / row["path"]
            true_label = LABEL_MAP[row["label"]]
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            with timer.measure():
                result = detector.detect_emotions(img, face_rectangles=[(0, 0, w, h)])
            if result:
                emotions = result[0]["emotions"]
                pred_label = max(emotions, key=emotions.get)
            else:
                pred_label = "none"
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
    print("peak VRAM/RAM (nvidia-smi used MB, TF may not report via CUDA alloc):", vram.peak_mb())
    print("summary written to", out)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)
