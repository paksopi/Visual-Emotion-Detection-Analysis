"""Method abstraction for the Track C harness: an arbitrary registered model
(src/eval/model_registry.py) run through an arbitrary registered evaluation
method, reusing the existing LatencyTimer/VRAMTracker/RunLogger/
classification_metrics primitives - not replacing them.

Existing bespoke runners (src/cv/run_*.py, src/vlm/run_*.py) stay the source
of truth for the 8 already-working models; `latency_vram` here is primarily
for NEW candidates that don't have a bespoke runner yet, and for cross-
checking the registry abstraction itself (see verification step 2 in the plan).
"""
import csv
import sys
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from eval import LatencyTimer, VRAMTracker, classification_metrics, RunLogger  # noqa: E402
from eval.model_registry import ModelAdapter  # noqa: E402
from eval import session_fitness as _session_fitness  # noqa: E402

FER2013_MANIFEST = REPO_ROOT / "data" / "fer2013" / "test_manifest_shuffled.csv"
FER2013_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
# Canonical FER2013 folder-name -> label mapping - a property of the dataset,
# identical across every run_*.py Track A runner (see e.g. run_hsemotion.py).
FER2013_DATASET_LABEL_MAP = {
    "angry": "angry", "disgusted": "disgust", "fearful": "fear", "happy": "happy",
    "sad": "sad", "surprised": "surprise", "neutral": "neutral",
}
TRACK_B_IMAGES_DIR = REPO_ROOT / "data" / "track_b" / "images"
TRACK_B_GROUND_TRUTH = REPO_ROOT / "data" / "track_b" / "ground_truth.csv"


class Method(Protocol):
    name: str
    applicable_output_kinds: set
    applicable_modalities: set

    def run(self, adapter: ModelAdapter, limit: Optional[int] = None) -> dict: ...


def _check_applicable(method: "Method", adapter: ModelAdapter) -> None:
    if adapter.output_kind not in method.applicable_output_kinds:
        raise ValueError(
            f"method '{method.name}' does not apply to output_kind='{adapter.output_kind}' "
            f"(model '{adapter.key}') - applicable kinds: {method.applicable_output_kinds}"
        )
    if adapter.modality not in method.applicable_modalities:
        raise ValueError(
            f"method '{method.name}' does not apply to modality='{adapter.modality}' "
            f"(model '{adapter.key}') - applicable modalities: {method.applicable_modalities}"
        )


class LatencyVramMethod:
    name = "latency_vram"
    applicable_output_kinds = {"closed_set_emotion", "native_other", "engagement", "free_text"}
    applicable_modalities = {"face_crop_bgr", "scene_rgb"}

    def run(self, adapter: ModelAdapter, limit: Optional[int] = None) -> dict:
        predict_fn = adapter.make_predictor()
        logger = RunLogger(model_name=adapter.key, track="A" if adapter.modality == "face_crop_bgr" else "B")
        timer = LatencyTimer()
        y_true, y_pred = [], []
        try:
            with VRAMTracker(use_torch=adapter.use_torch_vram) as vram:
                if adapter.modality == "face_crop_bgr":
                    n = self._run_face_crop(adapter, predict_fn, timer, logger, y_true, y_pred, limit)
                else:
                    n = self._run_scene_rgb(predict_fn, timer, logger, limit)
            summary = {
                "latency": timer.summary(),
                "peak_vram_mb": vram.peak_mb(),
                "n_images": n,  # matches the "n_images" field name every existing runner uses
                "license_id": adapter.license_id,
                "production_eligible": adapter.production_eligible,
            }
            if y_true:
                summary["metrics"] = classification_metrics(y_true, y_pred, labels=FER2013_LABELS)
            out = logger.write_summary(summary)
        finally:
            logger.close()
        return {"summary_path": str(out), **summary}

    def _run_face_crop(self, adapter, predict_fn, timer, logger, y_true, y_pred, limit):
        rows = list(csv.DictReader(open(FER2013_MANIFEST, encoding="utf-8")))
        if limit:
            rows = rows[:limit]
        for row in rows:
            img_path = REPO_ROOT / row["path"]
            img_bgr = cv2.imread(str(img_path))
            with timer.measure():
                pred = predict_fn(img_bgr)
            logger.record(image=str(img_path), pred=pred)
            if adapter.output_kind == "closed_set_emotion" and row["label"] in FER2013_DATASET_LABEL_MAP:
                y_true.append(FER2013_DATASET_LABEL_MAP[row["label"]])
                y_pred.append(pred if pred in FER2013_LABELS else pred)
        return len(rows)

    def _run_scene_rgb(self, predict_fn, timer, logger, limit):
        rows = list(csv.DictReader(open(TRACK_B_GROUND_TRUTH, encoding="utf-8")))
        if limit:
            rows = rows[:limit]
        for row in rows:
            image_id = row["image_id"]
            img = Image.open(TRACK_B_IMAGES_DIR / f"{image_id}.jpg").convert("RGB")
            frame_rgb = np.array(img)
            with timer.measure():
                answer = predict_fn(frame_rgb)
            logger.record(image_id=image_id, ground_truth=row["ground_truth"], raw_response=answer)
        return len(rows)


class Fer2013AccuracyMethod:
    name = "fer2013_accuracy"
    applicable_output_kinds = {"closed_set_emotion"}
    applicable_modalities = {"face_crop_bgr"}

    def run(self, adapter: ModelAdapter, limit: Optional[int] = None) -> dict:
        _check_applicable(self, adapter)
        predict_fn = adapter.make_predictor()
        rows = list(csv.DictReader(open(FER2013_MANIFEST, encoding="utf-8")))
        if limit:
            rows = rows[:limit]
        logger = RunLogger(model_name=adapter.key, track="A")
        timer = LatencyTimer()
        y_true, y_pred = [], []
        try:
            with VRAMTracker(use_torch=adapter.use_torch_vram) as vram:
                for row in rows:
                    img_bgr = cv2.imread(str(REPO_ROOT / row["path"]))
                    true_label = FER2013_DATASET_LABEL_MAP[row["label"]]
                    with timer.measure():
                        pred = predict_fn(img_bgr)
                    y_true.append(true_label)
                    y_pred.append(pred)
                    logger.record(image=row["path"], true=true_label, pred=pred)
            metrics = classification_metrics(y_true, y_pred, labels=FER2013_LABELS)
            summary = {
                "metrics": metrics, "latency": timer.summary(), "peak_vram_mb": vram.peak_mb(),
                "n_images": len(rows), "license_id": adapter.license_id,
                "production_eligible": adapter.production_eligible,
            }
            out = logger.write_summary(summary)
        finally:
            logger.close()
        return {"summary_path": str(out), **summary}


class SessionFitnessMethod:
    name = "session_fitness"
    applicable_output_kinds = {"closed_set_emotion", "native_other", "engagement", "free_text"}
    applicable_modalities = {"face_crop_bgr", "scene_rgb"}

    def run(self, adapter: ModelAdapter, limit: Optional[int] = None) -> dict:
        return _session_fitness.classify_model(adapter.key)


METHOD_REGISTRY: dict[str, Method] = {
    "latency_vram": LatencyVramMethod(),
    "fer2013_accuracy": Fer2013AccuracyMethod(),
    "session_fitness": SessionFitnessMethod(),
}
