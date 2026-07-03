"""Model registry for the Track C production-candidate comparison harness.

Generalizes the closure-factory pattern already used by the live webcam demos
(src/cv/live_webcam_demo.py, src/vlm/live_webcam_vlm_demo.py) so an arbitrary
method (src/eval/methods.py) can drive an arbitrary registered model without
N x M bespoke scripts. Every make_predictor() below is a thin, lazy wrapper
around an EXISTING factory in those two files - no model-loading code is
duplicated here.

modality:    "face_crop_bgr" (Track A CV models) | "scene_rgb" (Track B VLMs)
output_kind: "closed_set_emotion" | "native_other" | "engagement" | "free_text"
license_id:  key into src/eval/licenses.py::LICENSE_REGISTRY
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

PredictFn = Callable[[Any], Any]  # face_crop_bgr -> label str, or scene_rgb -> free text


@dataclass
class ModelAdapter:
    key: str
    display_name: str
    modality: str
    output_kind: str
    make_predictor: Callable[[], PredictFn]
    use_torch_vram: bool = True
    label_map: Optional[dict] = None
    license_id: str = "unknown"
    production_eligible: bool = True
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelAdapter] = {}


def register(adapter: ModelAdapter) -> None:
    MODEL_REGISTRY[adapter.key] = adapter


# ---- Track A / CV (wraps src/cv/live_webcam_demo.py factories) ----

register(ModelAdapter(
    key="fer_lib", display_name="fer (mini-xception)",
    modality="face_crop_bgr", output_kind="closed_set_emotion",
    make_predictor=lambda: __import__("cv.live_webcam_demo", fromlist=["make_fer_predictor"]).make_fer_predictor(),
    use_torch_vram=False, license_id="mit", production_eligible=True,
))

register(ModelAdapter(
    key="deepface", display_name="DeepFace",
    modality="face_crop_bgr", output_kind="closed_set_emotion",
    make_predictor=lambda: __import__("cv.live_webcam_demo", fromlist=["make_deepface_predictor"]).make_deepface_predictor(),
    use_torch_vram=False, license_id="mit", production_eligible=True,
))

register(ModelAdapter(
    key="hsemotion", display_name="HSEmotion / EmotiEffLib",
    modality="face_crop_bgr", output_kind="closed_set_emotion",
    make_predictor=lambda: __import__("cv.live_webcam_demo", fromlist=["make_hsemotion_predictor"]).make_hsemotion_predictor(),
    use_torch_vram=False, license_id="apache-2.0", production_eligible=True,
))

register(ModelAdapter(
    key="efficientface", display_name="EfficientFace",
    modality="face_crop_bgr", output_kind="closed_set_emotion",
    make_predictor=lambda: __import__("cv.live_webcam_demo", fromlist=["make_efficientface_predictor"]).make_efficientface_predictor(),
    use_torch_vram=True, license_id="mit", production_eligible=True,
    notes="Research code, no pip package - checkpoint fetched manually into data/models/efficientface/.",
))

register(ModelAdapter(
    key="mediapipe", display_name="MediaPipe (top blendshape, no emotion label)",
    modality="face_crop_bgr", output_kind="native_other",
    make_predictor=lambda: __import__("cv.live_webcam_demo", fromlist=["make_mediapipe_predictor"]).make_mediapipe_predictor(),
    use_torch_vram=False, license_id="apache-2.0", production_eligible=False,
    notes="UNLISTED - incapable: no built-in emotion label, outputs blendshapes/landmarks only.",
))

register(ModelAdapter(
    key="emotiefflib_engagement", display_name="EmotiEffLib engagement (sliding window)",
    modality="face_crop_bgr", output_kind="engagement",
    make_predictor=lambda: __import__("cv.live_webcam_demo", fromlist=["make_emotiefflib_engagement_predictor"]).make_emotiefflib_engagement_predictor(),
    use_torch_vram=False, license_id="apache-2.0", production_eligible=True,
    notes=("Directly targets student engagement/session state, not just categorical emotion - see "
           "ref/visual_emotion_detection_models.md §5. Per-frame feature extraction is realtime-class "
           "(~10ms, same backbone as HSEmotion); the windowed classify_engagement() call itself is "
           "expensive (~700ms, rebuilds+reloads a Keras model per call) so it's recomputed on a cadence "
           "in the live demo, not every frame - see make_emotiefflib_engagement_predictor()'s docstring. "
           "FER2013 has no video sequences, so fer2013_accuracy does not apply - see run_emotiefflib_engagement.py."),
))

# ---- Track B / VLM (wraps src/vlm/live_webcam_vlm_demo.py factories) ----
# Those factories return (predict_fn, model); the registry only needs predict_fn.

register(ModelAdapter(
    key="moondream2", display_name="Moondream2 (fp16)",
    modality="scene_rgb", output_kind="free_text",
    make_predictor=lambda: __import__("vlm.live_webcam_vlm_demo", fromlist=["make_moondream2_predictor"]).make_moondream2_predictor()[0],
    use_torch_vram=True, license_id="apache-2.0", production_eligible=True,
    notes="3-18s/call, sequential autoregressive decode - not viable as a per-turn real-time signal, see session_fitness.",
))

register(ModelAdapter(
    key="qwen25vl3b_4bit", display_name="Qwen2.5-VL-3B-Instruct (4-bit nf4)",
    modality="scene_rgb", output_kind="free_text",
    make_predictor=lambda: __import__("vlm.live_webcam_vlm_demo", fromlist=["make_qwen25vl_predictor"]).make_qwen25vl_predictor()[0],
    use_torch_vram=True, license_id="qwen-research", production_eligible=False,
    notes="UNLISTED - license: Qwen Research License, non-commercial. No numbers included in the unified comparison.",
))

register(ModelAdapter(
    key="moondream2_fast", display_name="Moondream2 (fast, one-word emotion)",
    modality="scene_rgb", output_kind="free_text",
    make_predictor=lambda: __import__("vlm.live_webcam_vlm_demo", fromlist=["make_moondream2_fast_predictor"]).make_moondream2_fast_predictor()[0],
    use_torch_vram=True, license_id="apache-2.0", production_eligible=True,
    notes=("Same model as 'moondream2', a short one-word-answer prompt instead of the reasoning prompt - "
           "measured ~860ms/call vs ~5s, close to Florence-2's native-captioning speed but with a real "
           "emotion label. Trade-off: no scene-grounded explanation, just the label - see "
           "src/vlm/live_webcam_vlm_demo.py SHORT_EMOTION_PROMPT note. Live A/B against qwen25vl3b_4bit_fast "
           "on the same 30s window (2026-07-03, user-confirmed not hallucinated): caught a real "
           "angry/disgusted expression change that qwen25vl3b_4bit_fast missed entirely (stayed flat "
           "'Neutral' the whole window) - more responsive, not just faster, at the cost of being visibly "
           "jumpier frame-to-frame."),
))

register(ModelAdapter(
    key="qwen25vl3b_4bit_fast", display_name="Qwen2.5-VL-3B-Instruct (fast, one-word emotion)",
    modality="scene_rgb", output_kind="free_text",
    make_predictor=lambda: __import__("vlm.live_webcam_vlm_demo", fromlist=["make_qwen25vl_fast_predictor"]).make_qwen25vl_fast_predictor()[0],
    use_torch_vram=True, license_id="qwen-research", production_eligible=False,
    notes=("UNLISTED - license: same model as 'qwen25vl3b_4bit', short one-word-answer prompt - "
           "measured ~1.2s/call vs ~10-17s, but still Qwen Research License regardless of speed."),
))

register(ModelAdapter(
    key="florence2", display_name="Florence-2-base (native captioning only)",
    modality="scene_rgb", output_kind="native_other",
    make_predictor=lambda: __import__("vlm.live_webcam_vlm_demo", fromlist=["make_florence2_predictor"]).make_florence2_predictor()[0],
    use_torch_vram=True, license_id="mit", production_eligible=False,
    notes=("UNLISTED - incapable: task-token model, not instruction-tuned - there is no free-text "
           "emotion prompt to set for it. Never scored on accuracy; no numbers included in the "
           "unified comparison."),
))

register(ModelAdapter(
    key="paligemma_mix_fast", display_name="PaliGemma-mix-224 (fast, one-word emotion)",
    modality="scene_rgb", output_kind="free_text",
    make_predictor=lambda: __import__("vlm.live_webcam_vlm_demo", fromlist=["make_paligemma_fast_predictor"]).make_paligemma_fast_predictor()[0],
    use_torch_vram=True, license_id="gemma", production_eligible=True,
    notes=("Instruction-tuned -mix-224 checkpoint (the raw -pt-224 pretrained checkpoint surveyed "
           "originally cannot be prompted at all - see ref/visual_emotion_detection_models.md). "
           "Short one-word-answer prompt only, same policy as moondream2_fast/qwen25vl3b_4bit_fast."),
))

register(ModelAdapter(
    key="minicpmv26_fast", display_name="MiniCPM-V 2.6 (fast, one-word emotion, 4-bit)",
    modality="scene_rgb", output_kind="free_text",
    make_predictor=lambda: (_ for _ in ()).throw(
        NotImplementedError("minicpmv26_fast: cancelled, no runner attempted - see notes")
    ),
    use_torch_vram=True, license_id="apache-2.0", production_eligible=False,
    notes=("UNLISTED - cancelled (2026-07-03, user call, local storage): its remote-code image "
           "processor (image_processing_minicpmv.py) raised a shape-mismatch error against this "
           "repo's pinned transformers==4.49.0 - a known class of incompatibility (MiniCPM-V-2_6's "
           "repo code targets an older transformers release), same pattern as Py-Feat's separate "
           ".venv-pyfeat. Would need its own isolated venv with an older transformers pin to run at "
           "all; the ~16GB fp16 checkpoint download was deleted to free local disk space and this "
           "candidate is not being pursued further. Revisit only if disk space and a dedicated venv "
           "are both available."),
))

register(ModelAdapter(
    key="llava15_7b", display_name="LLaVA-1.5-7B",
    modality="scene_rgb", output_kind="free_text",
    make_predictor=lambda: (_ for _ in ()).throw(
        NotImplementedError("llava15_7b: no runner built - excluded before any code was written, see notes")
    ),
    use_torch_vram=True, license_id="unknown", production_eligible=False,
    notes="UNLISTED - exceeds the 6GB VRAM budget outright. No runner built, no numbers collected.",
))

register(ModelAdapter(
    key="openface2", display_name="OpenFace 2.x",
    modality="face_crop_bgr", output_kind="native_other",
    make_predictor=lambda: (_ for _ in ()).throw(
        NotImplementedError("openface2: no runner built - excluded before any code was written, see notes")
    ),
    use_torch_vram=False, license_id="openface-academic", production_eligible=False,
    notes=("UNLISTED - incapable AND license-restricted: only outputs Action Units, no built-in "
           "emotion label, and CMU's academic/non-commercial license excludes it from shipping "
           "regardless. No runner built, no numbers collected."),
))

register(ModelAdapter(
    key="openface3", display_name="OpenFace 3.0",
    modality="face_crop_bgr", output_kind="closed_set_emotion",
    make_predictor=lambda: (_ for _ in ()).throw(
        NotImplementedError("openface3: no runner built - excluded before any code was written, see notes")
    ),
    use_torch_vram=False, license_id="openface3-academic", production_eligible=False,
    notes=("UNLISTED - license: adds a real emotion head (fixing 2.x's capability gap) but the "
           "CMU-MultiComp-Lab/OpenFace-3.0 LICENSE file is academic/non-profit non-commercial only, "
           "same terms as 2.x. Rejected at the license gate before any runner was built."),
))
