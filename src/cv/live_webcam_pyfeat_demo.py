"""Live webcam demo: Py-Feat (Detectorv1) against your own webcam feed.

Separate script from live_webcam_demo.py because Py-Feat needs its own venv
(.venv-pyfeat/) - its torchcodec dependency conflicts with the torch==2.6.0
pin the other Track A models share (see run_pyfeat.py docstring).

Feeds live frames directly as a tensor (data_type="tensor") instead of
writing to disk per frame - Detectorv1.detect's "image" mode only accepts
file paths.

Run: .venv-pyfeat/Scripts/python src/cv/live_webcam_pyfeat_demo.py [seconds]
Controls: 'q' quits early.
"""
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
os.add_dll_directory(str(REPO_ROOT / "bin" / "ffmpeg-shared"))
CASCADE_PATH = REPO_ROOT / "data" / "models" / "haarcascade_frontalface_default.xml"
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval import RunLogger  # noqa: E402

SECONDS = 15
MODEL_LABEL_MAP = {
    "anger": "angry", "disgust": "disgust", "fear": "fear", "happiness": "happy",
    "sadness": "sad", "surprise": "surprise", "neutral": "neutral",
}


def detect_face(gray, cascade):
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def crop_face(frame, box, pad=0.2):
    x, y, w, h = box
    H, W = frame.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    return frame[y0:y1, x0:x1]


def main():
    from feat.detector import Detectorv1

    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else SECONDS
    detector = Detectorv1(device="cuda" if torch.cuda.is_available() else "cpu")

    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam (index 0). Check camera permissions/connection.")
        sys.exit(1)

    logger = RunLogger(model_name="pyfeat_live", track="live-webcam")
    label_counts = Counter()
    n_frames = n_with_face = 0
    last_label = "..."
    window = "Live Emotion Demo - Py-Feat"

    print(f"Running Py-Feat for {seconds:.0f}s. Press 'q' to quit early.")
    start = time.time()
    try:
        while time.time() - start < seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            n_frames += 1
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            box = detect_face(gray, cascade)

            if box is not None:
                n_with_face += 1
                x, y, w, h = box
                face_rgb = cv2.cvtColor(crop_face(frame, box), cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(face_rgb.copy()).permute(2, 0, 1).unsqueeze(0)
                try:
                    result = detector.detect(tensor, data_type="tensor", progress_bar=False)
                    emotions = result.emotions
                    label = (
                        MODEL_LABEL_MAP.get(emotions.iloc[0].idxmax())
                        if not emotions.empty and not emotions.iloc[0].isna().all()
                        else None
                    )
                except Exception as e:
                    label = f"error: {e}"
                if label:
                    last_label = label
                    label_counts[label] += 1
                logger.record(elapsed_s=round(time.time() - start, 3), face_detected=True, label=label)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
                cv2.putText(frame, str(last_label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 220, 0), 2)
            else:
                logger.record(elapsed_s=round(time.time() - start, 3), face_detected=False, label=None)
                cv2.putText(frame, "no face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            remaining = seconds - (time.time() - start)
            cv2.putText(frame, f"Py-Feat  |  {remaining:0.1f}s left  |  q = quit",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        out = logger.write_summary({
            "seconds": seconds, "n_frames": n_frames,
            "n_frames_with_face": n_with_face, "label_counts": dict(label_counts),
        })
        logger.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"summary written to {out}")


if __name__ == "__main__":
    main()
