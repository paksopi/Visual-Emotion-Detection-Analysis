"""Live webcam demo: runs each Track A CV/FER model in turn against your own
webcam feed, drawing the detected face box + predicted label on screen, for
SECONDS_PER_MODEL seconds per model.

Uses the same inference calls as the batch runners (run_fer.py,
run_deepface.py, run_hsemotion.py, run_efficientface.py) so results are
directly comparable to the FER2013 benchmark numbers in
reports/model_comparison_results.md - only the face source changes (live
webcam + Haar-cascade crop instead of pre-cropped FER2013 images).

MediaPipe has no emotion label (see run_mediapipe.py) so its "label" here is
the single most active blendshape instead - not a real emotion prediction.

Py-Feat needs its own venv (torch/torchcodec conflict, see run_pyfeat.py) so
it isn't in this script - use live_webcam_pyfeat_demo.py instead. The VLMs
(Moondream2, Qwen2.5-VL, Florence-2) need whole-scene frames, not face crops,
and take seconds per call - see live_webcam_vlm_demo.py.

Run: .venv/Scripts/python src/cv/live_webcam_demo.py [seconds_per_model]

Controls: 'q' quits the whole demo early. Each model otherwise gets its
full time slice before moving to the next.

Logging: every frame's prediction is written live to results/logs/
via the same RunLogger the batch runners use (track="live-webcam"), plus
a label-distribution summary per model in results/eval/ once its slice
ends. There's no ground truth for a live face, so no accuracy metrics -
just the raw prediction stream and counts.
"""
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
CASCADE_PATH = REPO_ROOT / "data" / "models" / "haarcascade_frontalface_default.xml"
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval import RunLogger  # noqa: E402

SECONDS_PER_MODEL = 15


def detect_face(gray, cascade):
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])  # largest face


def crop_face(frame, box, pad=0.2):
    x, y, w, h = box
    H, W = frame.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    return frame[y0:y1, x0:x1]


# ---- per-model predictors (mirror the corresponding batch runner's calls) ----

def make_fer_predictor():
    from fer.fer import FER

    detector = FER(mtcnn=False, cascade_file=str(CASCADE_PATH))

    def predict(face_bgr):
        h, w = face_bgr.shape[:2]
        result = detector.detect_emotions(face_bgr, face_rectangles=[(0, 0, w, h)])
        if not result:
            return None
        emotions = result[0]["emotions"]
        return max(emotions, key=emotions.get)

    return predict


def make_deepface_predictor():
    from deepface import DeepFace

    def predict(face_bgr):
        result = DeepFace.analyze(
            face_bgr,
            actions=["emotion"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )
        return result[0]["dominant_emotion"]

    return predict


def make_hsemotion_predictor():
    import urllib.request  # noqa: F401 - hsemotion_onnx bug workaround, see run_hsemotion.py

    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

    recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")
    label_map = {
        "Anger": "angry",
        "Contempt": "contempt",
        "Disgust": "disgust",
        "Fear": "fear",
        "Happiness": "happy",
        "Neutral": "neutral",
        "Sadness": "sad",
        "Surprise": "surprise",
    }

    def predict(face_bgr):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        emotion, _scores = recognizer.predict_emotions(face_rgb, logits=False)
        return label_map.get(emotion, emotion)

    return predict


def make_efficientface_predictor():
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms

    efficientface_repo = REPO_ROOT / "data" / "models" / "efficientface_repo"
    if str(efficientface_repo) not in sys.path:
        sys.path.insert(0, str(efficientface_repo))
    from models.EfficientFace import efficient_face

    checkpoint = REPO_ROOT / "data" / "models" / "efficientface" / "rafdb.pth"
    idx_to_label = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = efficient_face()
    model.fc = nn.Linear(1024, 7)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = {k.replace("module.", "", 1): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    normalize = transforms.Normalize(
        mean=[0.57535914, 0.44928582, 0.40079932],
        std=[0.20735591, 0.18981615, 0.18132027],
    )
    preprocess = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )

    def predict(face_bgr):
        img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        x = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        idx = int(logits.argmax(dim=1).item())
        return idx_to_label[idx]

    return predict


def make_mediapipe_predictor():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = REPO_ROOT / "data" / "models" / "mediapipe" / "face_landmarker.task"
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, output_face_blendshapes=True, num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    def predict(face_bgr):
        # No emotion label (see run_mediapipe.py docstring) - report the
        # single most active blendshape instead, as a stand-in "label".
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
        result = detector.detect(mp_img)
        if not result.face_blendshapes:
            return None
        top = max(result.face_blendshapes[0], key=lambda c: c.score)
        return f"{top.category_name} ({top.score:.2f})"

    return predict


def make_emotiefflib_engagement_predictor():
    """Feature extraction (extract_features) is fast, ~10ms - see
    run_emotiefflib_engagement.py. classify_engagement() is NOT: it rebuilds
    a Keras model and reloads its weights from disk on every call (measured
    ~700ms, see that script's docstring/output) - calling it every frame
    would make the live demo look frozen. Recompute on a cadence instead
    (every RECOMPUTE_EVERY frames, ~1-2x/sec at typical webcam fps) and
    return the cached last score in between - engagement is a coarser,
    slower-moving signal than per-frame emotion anyway, so this doesn't lose
    anything meaningful.
    """
    import numpy as np
    from collections import deque
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    recognizer = EmotiEffLibRecognizer(engine="onnx", model_name="enet_b0_8_best_vgaf")
    window_width = 128
    buffer = deque(maxlen=window_width + 20)
    RECOMPUTE_EVERY = 15
    state = {"last": None, "frames_since_recompute": 0}

    def predict(face_bgr):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        features = recognizer.extract_features(face_rgb)[0]
        buffer.append(features)
        if len(buffer) <= window_width:
            return f"buffering ({len(buffer)}/{window_width + 1})"
        state["frames_since_recompute"] += 1
        if state["last"] is None or state["frames_since_recompute"] >= RECOMPUTE_EVERY:
            preds, _scores = recognizer.classify_engagement(
                np.stack(buffer), sliding_window_width=window_width
            )
            state["last"] = preds[-1]
            state["frames_since_recompute"] = 0
        return state["last"]

    return predict


MODELS = [
    ("fer (mini-xception)", make_fer_predictor),
    ("DeepFace", make_deepface_predictor),
    ("HSEmotion", make_hsemotion_predictor),
    ("EfficientFace", make_efficientface_predictor),
    ("MediaPipe (top blendshape, no emotion label)", make_mediapipe_predictor),
    ("EmotiEffLib engagement (sliding window)", make_emotiefflib_engagement_predictor),
]


def run_model_live(cap, cascade, model_name, predict_fn, seconds, logger):
    window = "Live Emotion Demo"
    start = time.time()
    last_label = "..."
    label_counts = Counter()
    n_frames = 0
    n_with_face = 0
    while True:
        elapsed = time.time() - start
        remaining = seconds - elapsed
        if remaining <= 0:
            break

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
            face = crop_face(frame, box)
            try:
                label = predict_fn(face)
            except Exception as e:
                label = f"error: {e}"
            if label:
                last_label = label
                label_counts[label] += 1
            logger.record(elapsed_s=round(elapsed, 3), face_detected=True, label=label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(frame, last_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 220, 0), 2)
        else:
            logger.record(elapsed_s=round(elapsed, 3), face_detected=False, label=None)
            cv2.putText(frame, "no face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"{model_name}  |  {remaining:0.1f}s left  |  q = quit",
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False, n_frames, n_with_face, label_counts
    return True, n_frames, n_with_face, label_counts


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else SECONDS_PER_MODEL

    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        print(f"Failed to load Haar cascade at {CASCADE_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam (index 0). Check camera permissions/connection.")
        sys.exit(1)

    print(f"Running {len(MODELS)} models, {seconds:.0f}s each. Press 'q' to quit early.")
    try:
        for model_name, make_predictor in MODELS:
            print(f"\n=== {model_name} === loading...")
            try:
                predict_fn = make_predictor()
            except Exception as e:
                print(f"Skipping {model_name}: failed to load ({e})")
                continue
            print(f"=== {model_name} === running for {seconds:.0f}s")
            model_key = model_name.split(" ")[0].lower()
            logger = RunLogger(model_name=f"{model_key}_live", track="live-webcam")
            try:
                keep_going, n_frames, n_with_face, label_counts = run_model_live(
                    cap, cascade, model_name, predict_fn, seconds, logger
                )
                out = logger.write_summary({
                    "seconds": seconds,
                    "n_frames": n_frames,
                    "n_frames_with_face": n_with_face,
                    "label_counts": dict(label_counts),
                })
                print(f"log: {logger.log_path}")
                print(f"summary written to {out}")
            finally:
                logger.close()
            if not keep_going:
                print("Quit requested.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
