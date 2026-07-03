"""Track A "native capability" runner: EmotiEffLib's predict_engagement().

Unlike predict_emotions() (already exercised via hsemotion-onnx in
run_hsemotion.py), engagement is a SEQUENCE model: it extracts a 2560-dim
feature vector per frame (same EfficientNet backbone as the emotion head),
then runs a TensorFlow/Keras attention classifier over a sliding window of
frames (default 128) to produce one engagement score - see
emotiefflib/engagement_classification_model.py. FER2013 is a bag of
unrelated static images, not a video sequence of one subject, so there is no
meaningful ground truth (or even meaningful *input*) for engagement here -
same "native capability, not a scorable benchmark" situation as
run_mediapipe.py.

What this script actually measures, split into its two real components:
1. Per-frame feature extraction latency (extract_features()) - this IS a
   real, real-time-relevant number: it's the per-call cost a live session
   would pay every frame regardless of engagement scoring.
2. The engagement-classification call's own latency (classify_engagement()),
   measured on a SYNTHETIC buffer (the first `sliding_window_width` FER2013
   feature vectors concatenated, not a real temporal sequence from one
   subject) - this characterizes the classification step's compute cost in
   isolation, not a meaningful engagement score. Do not read anything into
   the actual predicted classes here.

See ref/visual_emotion_detection_models.md §5 for the full onboarding
rationale (this is the single most use-case-relevant finding of that spike -
student engagement is closer to the actual target signal than 7/8-class
categorical emotion).
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval import LatencyTimer, VRAMTracker, RunLogger  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "fer2013" / "test_manifest_shuffled.csv"
SLIDING_WINDOW_WIDTH = 128
# classify_engagement() requires STRICTLY MORE frames than the window width
# (it slides across max_iters = n_frames - window_width positions; exactly
# window_width frames gives max_iters=0, an empty slice, and a confusing
# "Unsupported feature vector dim" error, not a clean "not enough frames"
# one) - collect a bit more so the synthetic call actually exercises the slide.
SYNTHETIC_BUFFER_SIZE = SLIDING_WINDOW_WIDTH + 20


def main(limit: int | None = None):
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    recognizer = EmotiEffLibRecognizer(engine="onnx", model_name="enet_b0_8_best_vgaf")

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    if len(rows) < SYNTHETIC_BUFFER_SIZE:
        print(f"warning: only {len(rows)} rows, need > {SLIDING_WINDOW_WIDTH} for the "
              f"synthetic-window classification-latency measurement; that part will be skipped")

    logger = RunLogger(model_name="emotiefflib_engagement", track="A-native")
    feature_timer = LatencyTimer()
    n_face = 0
    collected_features = []

    with VRAMTracker(use_torch=False) as vram:
        for i, row in enumerate(rows):
            img_path = REPO_ROOT / row["path"]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            with feature_timer.measure():
                features = recognizer.extract_features(img_rgb)
            n_face += 1
            if len(collected_features) < SYNTHETIC_BUFFER_SIZE:
                collected_features.append(features[0])
            logger.record(image=str(img_path), label=row["label"], feature_dim=int(features.shape[-1]))
            if i % 500 == 0:
                print(f"{i}/{len(rows)}")

    engagement_call_ms = None
    engagement_call_note = "skipped - not enough rows"
    if len(collected_features) >= SYNTHETIC_BUFFER_SIZE:
        synthetic_window = np.stack(collected_features[:SYNTHETIC_BUFFER_SIZE])  # (>128, feature_dim)
        t0 = time.perf_counter()
        preds, scores = recognizer.classify_engagement(synthetic_window, sliding_window_width=SLIDING_WINDOW_WIDTH)
        engagement_call_ms = (time.perf_counter() - t0) * 1000
        engagement_call_note = (
            "SYNTHETIC input (first 128 unrelated FER2013 images concatenated as a fake "
            "'sequence') - measures the classification call's own compute cost only, the "
            "resulting scores are not a meaningful engagement read"
        )
        print(f"engagement classify() call: {engagement_call_ms:.1f}ms on synthetic window, raw scores shape={scores.shape}")

    summary = {
        "capability": "predict_engagement() - sliding-window (default 128 frames) attention classifier over per-frame EfficientNet features, binary engaged/disengaged",
        "n_images": len(rows),
        "n_features_extracted": n_face,
        "peak_vram_mb": vram.peak_mb(),
        # This is the real-time-relevant number session_fitness reads: the
        # per-frame cost every live session pays, regardless of when/whether
        # the windowed engagement score is computed on top.
        "latency": feature_timer.summary(),
        "sliding_window_width": SLIDING_WINDOW_WIDTH,
        "engagement_classification_call_ms": engagement_call_ms,
        "engagement_classification_note": engagement_call_note,
        "note": (
            "Not an emotion/engagement accuracy benchmark - FER2013 has no video sequences, "
            "so there is no meaningful ground truth (or meaningful input) for a sliding-window "
            "engagement model here. 'latency' above is genuine per-frame feature-extraction cost "
            "(real-time-relevant); the engagement classification call itself was only measured "
            "for its own compute cost on a synthetic buffer, see engagement_classification_note. "
            "A real deployment needs a live per-subject frame buffer (e.g. the last few seconds "
            "of webcam frames), not single unrelated images."
        ),
    }
    out_path = logger.write_summary(summary)
    logger.close()
    print("feature extraction latency:", summary["latency"])
    print("summary written to", out_path)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)
