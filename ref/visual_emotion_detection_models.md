# Visual Emotion Detection Models for Perception Layer

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
| **EmotiEffLib (HSEmotion)**| CV / EfficientNet| **~1 GB** (PyTorch / ONNX) | [HSE-asavchenko/face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition) | ✅ Yes — `predict_emotions()` returns categorical label (8 classes) + scores directly |
| **EfficientFace** | CV / ResNet | **~1 GB** | [zengqunzhao/EfficientFace](https://github.com/zengqunzhao/EfficientFace) | ✅ Yes, with caveat — pretrained heads on RAF-DB/AffectNet give direct 7-class output, but it's research code (manual checkpoint download, no pip package) |
| **Py-Feat** | CV / Toolkit | **~1 GB - 1.5 GB** | [cosanlab/py-feat](https://github.com/cosanlab/py-feat) | ✅ Yes — `Detector` output includes a dedicated `fex.emotions` field alongside Action Units, not AUs alone |
| **OpenFace** | CV / Toolkit | **~1 GB - 2 GB** (Tracks facial muscle Action Units) | [TadasBaltrusaitis/OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) | ⚠️ **No** — outputs only Action Unit presence/intensity, no direct emotion classification; users must do their own FACS→emotion mapping |

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
