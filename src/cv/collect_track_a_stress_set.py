"""Track A self-collected occlusion/lighting stress test.

FER2013 images are clean, pre-cropped, front-on 48x48 faces - none of the
four Track A models have ever been tested against the conditions a real
perception layer actually sees: dim lighting, partial occlusion, off-angle
faces. This script captures a few real webcam frames under those conditions,
saves them to data/track_a_stress/<condition>/, then runs all four Track A
models (fer, DeepFace, HSEmotion, EfficientFace) against every captured
frame.

There is no emotion ground truth here - this isn't an accuracy benchmark,
it's a robustness check. The question is which models keep working (detect
a face at all, return a plausible/stable label) as conditions degrade from
the FER2013 norm, not which one is "correct."

Run: .venv/Scripts/python src/cv/collect_track_a_stress_set.py

During capture, follow the on-screen instructions for each condition; frames
are captured automatically on a timer. Press 'q' at any time to abort.
"""
import json
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
CASCADE_PATH = REPO_ROOT / "data" / "models" / "haarcascade_frontalface_default.xml"
STRESS_DATA_DIR = REPO_ROOT / "data" / "track_a_stress"
REPORT_PATH = REPO_ROOT / "reports" / "track_a_stress_test_results.md"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_webcam_demo import (  # noqa: E402
    crop_face,
    detect_face,
    make_deepface_predictor,
    make_efficientface_predictor,
    make_fer_predictor,
    make_hsemotion_predictor,
)

CONDITIONS = [
    ("baseline", "Normal lighting, face fully visible, look at the camera"),
    ("dim_light", "Turn down / dim the room lighting"),
    ("occluded", "Partially cover part of your face (hand, mask, hair)"),
    ("off_angle", "Turn your head to a 3/4 profile angle"),
]
FRAMES_PER_CONDITION = 3
SECONDS_BETWEEN_FRAMES = 3

MODEL_FACTORIES = [
    ("fer", make_fer_predictor),
    ("deepface", make_deepface_predictor),
    ("hsemotion", make_hsemotion_predictor),
    ("efficientface", make_efficientface_predictor),
]


def collect(cap, cascade):
    window = "Track A Stress Set Collection"
    STRESS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for condition, instruction in CONDITIONS:
        cond_dir = STRESS_DATA_DIR / condition
        cond_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {condition} === {instruction}")
        next_capture = time.time() + SECONDS_BETWEEN_FRAMES
        captured = 0
        while captured < FRAMES_PER_CONDITION:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            remaining = next_capture - time.time()

            display = frame.copy()
            cv2.putText(display, instruction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.putText(display, f"{condition}: frame {captured + 1}/{FRAMES_PER_CONDITION} "
                                  f"in {max(0, remaining):0.1f}s  |  q = abort",
                        (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 220, 0), 2)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyWindow(window)
                return manifest, False

            if remaining <= 0:
                out_path = cond_dir / f"{captured:02d}.jpg"
                cv2.imwrite(str(out_path), frame)
                manifest.append({"condition": condition, "path": str(out_path)})
                print(f"  captured {out_path.name}")
                captured += 1
                next_capture = time.time() + SECONDS_BETWEEN_FRAMES

    cv2.destroyWindow(window)
    return manifest, True


def evaluate(manifest, cascade):
    predictors = {}
    for name, factory in MODEL_FACTORIES:
        print(f"loading {name}...")
        predictors[name] = factory()

    results = []
    for row in manifest:
        img = cv2.imread(row["path"])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box = detect_face(gray, cascade)
        face_detected = box is not None
        preds = {}
        if face_detected:
            face = crop_face(img, box)
            for name, predict_fn in predictors.items():
                try:
                    preds[name] = predict_fn(face)
                except Exception as e:
                    preds[name] = f"error: {e}"
        results.append({
            "condition": row["condition"],
            "path": row["path"],
            "face_detected": face_detected,
            "predictions": preds,
        })
    return results


def write_report(results):
    conditions = [c for c, _ in CONDITIONS]
    model_names = [name for name, _ in MODEL_FACTORIES]

    lines = [
        "# Track A — self-collected occlusion/lighting stress test",
        "",
        "Robustness check, not an accuracy benchmark: real webcam frames under "
        "conditions FER2013 doesn't cover (dim lighting, partial occlusion, "
        "off-angle), with no emotion ground truth. Reports face-detection "
        "success and predicted label per model per condition.",
        "",
        f"Frames per condition: {FRAMES_PER_CONDITION}. Conditions: "
        + ", ".join(conditions) + ".",
        "",
        "## Face-detection rate by condition",
        "",
        "| Condition | Frames | Face detected |",
        "|---|---|---|",
    ]
    for condition in conditions:
        rows = [r for r in results if r["condition"] == condition]
        n = len(rows)
        n_detected = sum(1 for r in rows if r["face_detected"])
        lines.append(f"| {condition} | {n} | {n_detected}/{n} |")

    lines += ["", "## Predicted label by model per condition (face-detected frames only)", ""]
    lines.append("| Condition | " + " | ".join(model_names) + " |")
    lines.append("|---|" + "---|" * len(model_names))
    for condition in conditions:
        rows = [r for r in results if r["condition"] == condition and r["face_detected"]]
        cells = []
        for name in model_names:
            labels = [r["predictions"].get(name, "-") for r in rows]
            cells.append(", ".join(labels) if labels else "no face detected")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")

    lines += ["", "## Raw per-frame results", "", "```json",
              json.dumps(results, indent=2), "```", ""]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport written to {REPORT_PATH}")


def main():
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        print(f"Failed to load Haar cascade at {CASCADE_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam (index 0).")
        sys.exit(1)

    try:
        manifest, completed = collect(cap, cascade)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not manifest:
        print("No frames captured, aborting.")
        return
    if not completed:
        print(f"Aborted early - evaluating the {len(manifest)} frame(s) captured so far.")

    results = evaluate(manifest, cascade)
    write_report(results)


if __name__ == "__main__":
    main()
