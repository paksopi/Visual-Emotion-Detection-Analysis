"""Track A runner: HSEmotion/EmotiEffLib (enet_b0_8_best_vgaf, 8-class ONNX).

Note: this model outputs 8 classes (adds "Contempt" over FER2013's 7). FER2013
ground truth has no "contempt" label, so a "Contempt" prediction is always
scored as a miss against the 7-class FER2013 labels below — that's a fair
comparison, not a harness bug (it reflects a real train/eval label-set
mismatch this model would have in production against FER2013-style ground
truth).

Also works around a real bug in hsemotion_onnx: it calls urllib.request
without importing the submodule, which raises `AttributeError: module
'urllib' has no attribute 'request'` unless something else already imported
it — importing urllib.request first avoids the crash.
"""
import csv
import sys
import urllib.request  # noqa: F401 - see docstring; import is the fix
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

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
MODEL_LABEL_MAP = {
    "Anger": "angry",
    "Contempt": "contempt",  # not in FER2013 label set -> always a miss
    "Disgust": "disgust",
    "Fear": "fear",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprise",
}
MANIFEST = REPO_ROOT / "data" / "fer2013" / "test_manifest_shuffled.csv"


def main(limit: int | None = None):
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

    recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    if limit:
        rows = rows[:limit]

    logger = RunLogger(model_name="hsemotion", track="A")
    timer = LatencyTimer()
    y_true, y_pred = [], []

    with VRAMTracker(use_torch=False) as vram:
        for i, row in enumerate(rows):
            img_path = REPO_ROOT / row["path"]
            true_label = DATASET_LABEL_MAP[row["label"]]
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with timer.measure():
                emotion, _scores = recognizer.predict_emotions(img_rgb, logits=False)
            pred_label = MODEL_LABEL_MAP[emotion]
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
        "note": "model has an 8th class (contempt) not present in FER2013 ground truth",
    }
    out = logger.write_summary(summary)
    logger.close()
    print("accuracy:", metrics["accuracy"], "macro_f1:", metrics["macro_f1"])
    print("latency:", timer.summary())
    print("summary written to", out)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)
