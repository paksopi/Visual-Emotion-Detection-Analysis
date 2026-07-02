"""Track A runner: Py-Feat (Detectorv1, retinaface + xgboost emotion head).

Runs in its own venv (.venv-pyfeat/), not the shared .venv/ every other Track
A model uses: py-feat's video-decoding dependency (torchcodec) requires
torch>=2.11, which conflicts with the torch==2.6.0 pin the rest of this repo's
models need. See reports/model_comparison_results.md Sec 1 for the full
diagnosis (it was a torch/torchcodec version mismatch, not really an FFmpeg
problem, though a shared-library FFmpeg build -- bin/ffmpeg-shared/, gitignored
-- is also required on Windows since the system FFmpeg here is static-only).

Direct 7-class emotion output (anger/disgust/fear/happiness/sadness/surprise/
neutral) with no extra classes, unlike HSEmotion's 8-class output.
"""
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.add_dll_directory(str(REPO_ROOT / "bin" / "ffmpeg-shared"))
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
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}
MANIFEST = REPO_ROOT / "data" / "fer2013" / "test_manifest_shuffled.csv"


def main(limit: int | None = None):
    from feat.detector import Detectorv1

    detector = Detectorv1(device="cuda")

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    if limit:
        rows = rows[:limit]

    logger = RunLogger(model_name="pyfeat", track="A")
    timer = LatencyTimer()
    y_true, y_pred = [], []
    n_no_face = 0

    with VRAMTracker(use_torch=True) as vram:
        for i, row in enumerate(rows):
            img_path = REPO_ROOT / row["path"]
            true_label = DATASET_LABEL_MAP[row["label"]]
            with timer.measure():
                result = detector.detect([str(img_path)], data_type="image", progress_bar=False)
            emotions = result.emotions
            if emotions.empty or emotions.iloc[0].isna().all():
                n_no_face += 1
                logger.record(image=str(img_path), true=true_label, pred=None, no_face=True)
                continue
            pred_label = MODEL_LABEL_MAP[emotions.iloc[0].idxmax()]
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
        "n_no_face_detected": n_no_face,
    }
    out = logger.write_summary(summary)
    logger.close()
    print("accuracy:", metrics["accuracy"], "macro_f1:", metrics["macro_f1"])
    print("no-face count:", n_no_face, "/", len(rows))
    print("latency:", timer.summary())
    print("summary written to", out)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)
