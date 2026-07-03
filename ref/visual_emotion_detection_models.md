# Visual Emotion Detection Models for Perception Layer

## Selection criteria

Every candidate model - already in this survey, or found later via a link/search - is checked
against these in order. Failing any one is a hard exclusion regardless of the others; the model
goes straight to the Unlisted models table (§6) with no benchmark run and no numbers anywhere in
this repo:

1. **Capability** - must produce an emotion/affect signal directly, not an adjacent output
   (blendshapes, Action Units) with no label.
2. **License** - commercial-use compatible (MIT/Apache-2.0/etc.). Research-only or
   non-commercial licenses are excluded regardless of accuracy.
3. **VRAM** - must fit inside the 6GB budget shared with the tutoring LLM itself, measured
   peak, not vendor-quoted.
4. **Latency** - real-time (<=300ms, synchronous), borderline (debounced async), or async-only
   (periodic enrichment, never gates a turn) per `src/eval/session_fitness.py`.
5. **Accuracy** - measured on the unified sample (`data/unified_eval/`, see
   `reports/unified_comparison.md`), same metric for every model regardless of type.

## Executive Summary
This document outlines 14 open-source models suitable for integrating visual emotion detection into local perception layers. The models are evaluated against a strict 6GB VRAM hardware constraint and are divided into two categories: ultra-lightweight Computer Vision (CV) models and deeper Vision-Language Models (VLMs).

## 1. Traditional ML / Computer Vision Models (Lightweight)
These models are highly optimized for fast facial expression recognition. They operate in milliseconds and require cropping the face from the webcam frame prior to inference. They output categorical emotion labels (e.g., Happy, Sad, Neutral) rather than contextual reasoning.

| Model Name | Type | Estimated VRAM Needed | Official Repository / Link | Direct Emotion Output (verified) |
| :--- | :--- | :--- | :--- | :--- |
| **Mini-Xception (FER2013)** | CV / CNN | **< 100 MB** (Can run entirely on CPU) | [oarriaga/face_classification](https://github.com/oarriaga/face_classification) | ✅ Yes — ships trained FER2013 weights, 7-class emotion output built in |
| **MediaPipe** | CV / Tracking | **< 100 MB** (CPU/Mobile optimized) | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) | ⚠️ **No** — stock Face Landmarker only outputs 52 blendshape scores + raw landmarks, no built-in emotion label; needs a separate classifier trained on top |
| **fer** | CV / Python Lib | **< 1 GB** | [justinshenk/fer](https://github.com/justinshenk/fer) | ✅ Yes — `detect_emotions()` returns 7-class emotion dict with confidence scores directly |
| **DeepFace** | CV / Python Lib | **~1 GB** (Depends on backend e.g., VGG-Face) | [serengil/deepface](https://github.com/serengil/deepface) | ✅ Yes — `DeepFace.analyze(actions=['emotion'])` returns `dominant_emotion` + per-class scores directly |
| **EmotiEffLib (HSEmotion)**| CV / EfficientNet| **~1 GB** (PyTorch / ONNX) | [sb-ai-lab/EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) | ✅ Yes — `predict_emotions()` returns categorical label (8 classes) + scores directly. **Also ships `predict_engagement()`** (see §5) — a sliding-window (default 128 frames) attention classifier, separate signal from per-frame emotion, directly relevant to student-session monitoring |
| **EfficientFace** | CV / ResNet | **~1 GB** | [zengqunzhao/EfficientFace](https://github.com/zengqunzhao/EfficientFace) | ✅ Yes, with caveat — pretrained heads on RAF-DB/AffectNet give direct 7-class output, but it's research code (manual checkpoint download, no pip package) |
| **Py-Feat** | CV / Toolkit | **~1 GB - 1.5 GB** | [cosanlab/py-feat](https://github.com/cosanlab/py-feat) | ✅ Yes — `Detector` output includes a dedicated `fex.emotions` field alongside Action Units, not AUs alone |
| **OpenFace** | CV / Toolkit | **~1 GB - 2 GB** (Tracks facial muscle Action Units) | [TadasBaltrusaitis/OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) | ⚠️ **No** (2.x) — outputs only Action Unit presence/intensity, no direct emotion classification; users must do their own FACS→emotion mapping. **OpenFace 3.0 adds a direct emotion-recognition head (see §5) but is non-commercial-licensed** — not usable for a commercial product regardless of capability |

## 2. Vision-Language Models (VLMs)
These models analyze the entire scene, providing reasoning and context behind the user's emotional state (e.g., body language, environment). They are significantly heavier and will consume a large portion of a 6GB VRAM budget, reducing concurrency headroom for other models.

| Model Name | Type | Estimated VRAM Needed | Official Repository / Link | Emotion/Affect Reasoning (verified) |
| :--- | :--- | :--- | :--- | :--- |
| **Florence-2** | VLM (0.2B - 0.7B) | **0.9 GB - 1.5 GB** | [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) | ⚠️ **No, wrong checkpoint** — `-base` is the raw task-token model (`<CAPTION>`, `<OD>`, `<VQA>`), not an instruction/chat model; it can't reliably answer an open-ended "what emotion is this?" prompt without fine-tuning first |
| **Qwen2.5-VL-3B** | VLM (3B) | **2.0 GB - 3.0 GB** (Using 4-bit Quantization) | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | ✅ Yes, with caveat — instruction-tuned, does free-form VQA; EEmo-Bench shows real but coarse-grained emotion understanding (good on valence, weaker on arousal), and the 3B variant is smaller/weaker than the larger Qwen2.5-VL sizes benchmarked |
| **Moondream2** | VLM (1.8B) | **3.0 GB - 4.0 GB** | [vikhyat/moondream](https://github.com/vikhyat/moondream) | ✅ Yes, with caveat — instruction-tuned VQA/captioning, can describe people/actions; no dedicated emotion benchmark found, so affect reasoning is inferred from general VQA ability, not directly demonstrated |
| **Gemma Multimodal (PaliGemma)**| VLM (3B) | **3.0 GB - 4.0 GB** (Using 4-bit Quantization) | [google/paligemma-3b-**mix**-224](https://huggingface.co/google/paligemma-3b-mix-224) | ⚠️ **No, wrong checkpoint linked in original survey** — the `-pt-224` checkpoint is the raw pretrained model, explicitly documented as not benchmark/user-friendly and unable to follow open-ended instructions at all; use the `-mix` (or a fine-tuned) checkpoint instead, corrected here |
| **MiniCPM-V 2.6** | VLM (8B) | **~5.5 GB - 6.0 GB** (Using 4-bit Quantization) | [OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) | ✅ Yes, with caveat — strong general instruction-tuned VLM; the emotion-specific evidence found (audio+vision affect handling) is documented for the sibling MiniCPM-**o**-2.6, not directly confirmed for the vision-only V-2.6 checkpoint linked here |
| **LLaVA-1.5-7B** | VLM (7B) | **> 6.0 GB** (Too heavy for a 6GB VRAM limit) | [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | ✅ Yes, with caveat — confirmed capable on emotion/facial-analysis benchmarks (FaceBench), but consistently the weakest performer of the group, and already over the VRAM budget |

## 3. Core Trade-off Summary
* **Go with CV Models if:** The system requires continuous, real-time background monitoring (high frame rate) and needs to leave the GPU entirely free for other primary processes.
* **Go with VLMs if:** The system triggers visual perception selectively (e.g., only when the user speaks) and requires deep, human-like understanding of *why* the user feels a certain way based on their environment.

## 4. Emotion-Capability Verification (2026-07-02)

Every model in this survey was re-checked against its official README/model card/docs to confirm
it actually produces an emotion output, rather than just being adjacent to the task. Result:
**10 of 14 confirmed capable out of the box (some with caveats); 4 need a fix before use.**

**CV/FER track — 6 of 8 confirmed as direct, built-in emotion classifiers.** Two are not:
* **MediaPipe** — is a face *tracking* framework. Its stock Face Landmarker returns blendshapes/landmarks only; every "MediaPipe emotion detection" example online is a third-party classifier trained on top of it. Usable in this survey only as the face-tracking front end for another model, not as the emotion classifier itself.
* **OpenFace** — outputs Action Units only, no emotion label. Confirmed by an open upstream issue asking the maintainers exactly this question, still unresolved. Same status as MediaPipe: useful as an AU feature extractor, not a standalone emotion classifier.

**VLM track — 4 of 6 confirmed capable (with caveats); 2 need a different checkpoint.**
* **Florence-2** — the linked `-base` checkpoint is a task-token model (`<CAPTION>`, `<OD>`, `<VQA>`), not instruction-tuned, and can't reliably take an open-ended "what emotion is this?" prompt. Needs a fine-tuned or instruction-following variant to be usable for this task at all.
* **PaliGemma** — the linked `-pt-224` checkpoint is the raw pretrained model, explicitly documented by Google as not meant for direct prompting. **Corrected above to `-mix-224`**, the instruction-tuned checkpoint that can actually answer open-ended questions.
* Qwen2.5-VL-3B, Moondream2, MiniCPM-V 2.6, and LLaVA-1.5-7B are all genuinely promptable for emotion/affect reasoning, but none has emotion-specific benchmarking as strong as the CV track's dedicated classifiers — treat their VLM scores as "can reason about it," not "measured accurate at it," until Track B evaluation (see `reports/evaluation_plan.md`) actually runs.

## 5. 2026-07-03 spike: two new candidates checked for the production-selection harness

Checked while scoping `reports/production_candidate_comparison.md` (Track C, see
`reports/evaluation_plan.md`), specifically for a real-time student-emotion signal
usable in a commercial LLM-tutoring product:

* **OpenFace 3.0** ([CMU-MultiComp-Lab/OpenFace-3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0),
  arXiv [2506.02891](https://arxiv.org/abs/2506.02891)) — genuinely adds a direct
  FC-layer emotion-recognition head alongside landmarks/AUs/gaze in one lightweight
  multitask model, fixing what §4 above documents as OpenFace 2.x's core gap. **However,
  its `LICENSE` (fetched directly from the repo) is CMU's "ACADEMIC OR NON-PROFIT
  ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY" agreement** — the same restrictive
  lineage as the original OpenFace toolkit (non-commercial, ~$18k/yr for a commercial
  license per CMU's licensing portal). **Not built into this repo's harness** — excluded
  at the license gate before any runner code was written, per this repo's production-fit
  requirement. Revisit only if CMU changes terms or a commercial license is purchased.
* **EmotiEffLib engagement mode** — the library already used in this repo (as
  `hsemotion-onnx` 0.3.1, the older API) has a successor PyPI package, `emotiefflib`
  (checked: v1.1.1, **Apache-2.0, "no limitation for both academic and commercial
  usage"**), which adds `EmotiEffLibRecognizer.predict_engagement(face_imgs, sliding_window_width=128)`
  — a genuinely different signal from categorical emotion, more directly relevant to
  "is this student engaged/disengaged" than a 7-8 class emotion label. Mechanically: it
  extracts a 2560-dim feature vector per frame (same backbone as `predict_emotions`,
  `torch` or `onnx` engine), then runs a TensorFlow/Keras attention model over a
  **sliding window of frames (default 128)** — i.e. it needs several seconds of buffered
  frames, not a single frame, to produce one engagement score. TensorFlow was already a
  working dependency in this repo (via `fer`/DeepFace), confirmed at v2.21.0. **Cleared
  for onboarding** (`src/cv/run_emotiefflib_engagement.py`) — the single most
  use-case-relevant finding of this spike, since it targets session/engagement state
  directly rather than being a proxy via categorical emotion.

## 6. Unlisted models

Fail one or more of the selection criteria in the header above - no accuracy/latency numbers
exist for these anywhere in this repo (README, reports, `results/`). See
`src/eval/model_registry.py` for the authoritative `notes` field per model.

| Model | Failed criterion | Reason |
|---|---|---|
| MediaPipe | Capability | No built-in emotion label, blendshapes/landmarks only |
| OpenFace 2.x | Capability + License | Action Units only, no emotion label; CMU non-commercial license |
| OpenFace 3.0 | License | Adds a real emotion head, but same CMU academic/non-commercial license as 2.x |
| Florence-2-base | Capability | Task-token model, not instruction-tuned - no open-ended emotion prompt exists |
| Qwen2.5-VL-3B-Instruct | License | Qwen Research License, non-commercial |
| LLaVA-1.5-7B | VRAM | Exceeds the 6GB budget outright, no runner built |
| MiniCPM-V 2.6 | Cancelled | Clears capability/license/VRAM on paper, but its remote code is incompatible with this repo's pinned transformers==4.49.0; ~16GB checkpoint deleted to free disk space, not pursued further |

## 7. New candidates benchmarked this round (2026-07-03)

Added per user request, only in short/capped-token-output form (never a long reasoning prompt),
run through `unified_accuracy` (see `reports/unified_comparison.md`):

* **PaliGemma-mix-224** - clears capability/license/VRAM criteria. Benchmark run blocked in this
  environment: the checkpoint is a gated Hugging Face repo requiring an authenticated account
  that has accepted Google's Gemma terms. Pending re-run once HF auth is available.
* **MiniCPM-V 2.6** - clears capability/license (Apache-2.0 per model card, unconfirmed) and VRAM
  on paper. HF auth was resolved, but its remote-code image processor raised a shape-mismatch
  error against this repo's pinned `transformers==4.49.0` - a version-drift incompatibility, not
  an auth problem. **Cancelled** (2026-07-03, user call): the ~16GB fp16 checkpoint was deleted to
  free local disk space; not pursued further. Would need its own isolated venv (same pattern as
  Py-Feat's `.venv-pyfeat`) with an older transformers pin to ever run.
