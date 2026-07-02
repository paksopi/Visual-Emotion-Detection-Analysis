"""Track A "native capability" runner: MediaPipe FaceLandmarker.

MediaPipe has no built-in emotion label (confirmed in the survey doc, §4 /
README's Emotion-capability verification table) -- it only outputs face
landmarks and 52 ARKit-style blendshape coefficients (eyeBlinkLeft, jawOpen,
mouthSmileRight, etc). Forcing it into the Track A emotion-accuracy table
would just be "N/A" everywhere, so this runs it on its actual native task
instead and reports what it's proficient at: face-detection rate, per-frame
landmark/blendshape latency, and which blendshapes are most active on
average per FER2013 emotion-labeled folder (a proxy for whether its
blendshape geometry tracks emotion-adjacent facial movement at all, without
claiming it "detects emotion").
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval import LatencyTimer, RunLogger  # noqa: E402

MODEL_PATH = REPO_ROOT / "data" / "models" / "mediapipe" / "face_landmarker.task"
MANIFEST = REPO_ROOT / "data" / "fer2013" / "test_manifest_shuffled.csv"


def main(limit: int | None = None):
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    if limit:
        rows = rows[:limit]

    logger = RunLogger(model_name="mediapipe", track="A-native")
    timer = LatencyTimer()
    n_face_detected = 0
    blendshape_sums_by_label = defaultdict(lambda: defaultdict(float))
    blendshape_counts_by_label = defaultdict(int)

    for i, row in enumerate(rows):
        img_path = REPO_ROOT / row["path"]
        label = row["label"]
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        with timer.measure():
            result = detector.detect(mp_img)

        has_face = len(result.face_landmarks) > 0
        if has_face:
            n_face_detected += 1
            blendshape_counts_by_label[label] += 1
            for c in result.face_blendshapes[0]:
                blendshape_sums_by_label[label][c.category_name] += c.score
        logger.record(image=str(img_path), label=label, face_detected=has_face)
        if i % 500 == 0:
            print(f"{i}/{len(rows)}")

    top_blendshapes_by_label = {}
    for label, sums in blendshape_sums_by_label.items():
        n = blendshape_counts_by_label[label]
        means = {name: total / n for name, total in sums.items()}
        top5 = sorted(means.items(), key=lambda kv: -kv[1])[:5]
        top_blendshapes_by_label[label] = [{"blendshape": name, "mean_score": round(score, 4)} for name, score in top5]

    summary = {
        "capability": "face landmarks + 52 ARKit-style blendshape coefficients (no built-in emotion label)",
        "n_images": len(rows),
        "n_face_detected": n_face_detected,
        "face_detection_rate": n_face_detected / len(rows) if rows else 0.0,
        "latency": timer.summary(),
        "top5_blendshapes_by_fer2013_label": top_blendshapes_by_label,
        "note": (
            "This is not an emotion-accuracy benchmark -- MediaPipe has no emotion output. "
            "The per-label blendshape means show which facial-movement coefficients are most "
            "active on average for images from each FER2013 folder, as a proxy for whether its "
            "landmark geometry tracks emotion-adjacent expression at all."
        ),
    }
    out_path = logger.write_summary(summary)
    logger.close()
    print("face detection rate:", summary["face_detection_rate"])
    print("latency:", summary["latency"])
    print("summary written to", out_path)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)
