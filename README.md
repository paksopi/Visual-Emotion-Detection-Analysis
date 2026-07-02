# Visual Emotion Detection Model Survey

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A survey and evaluation plan for open-source visual emotion detection models, scoped for
integration into a local perception layer under a strict **6GB VRAM** hardware budget. 14
candidate models are documented, split into two families with very different characteristics:
lightweight facial-expression classifiers and heavier vision-language models (VLMs) that reason
about the whole scene.

No models have been installed or benchmarked yet — this repo currently captures the model survey
and the evaluation methodology that will be used once runs start.

- Model survey (14 candidates, VRAM estimates, links): [`ref/visual_emotion_detection_models.md`](ref/visual_emotion_detection_models.md)
- Evaluation plan (how each family will be measured): [`reports/evaluation_plan.md`](reports/evaluation_plan.md)

## The two model families

| Family | What it outputs | Examples | VRAM range |
|---|---|---|---|
| CV / FER (lightweight) | Categorical label (Happy, Sad, Neutral, ...) from a cropped face | Mini-Xception, MediaPipe, `fer`, DeepFace, EmotiEffLib, EfficientFace, Py-Feat, OpenFace | <100MB – ~2GB |
| VLM (contextual) | Free-text reasoning about the whole scene (body language, environment) | Florence-2, Qwen2.5-VL-3B, Moondream2, PaliGemma, MiniCPM-V 2.6, LLaVA-1.5-7B | 0.9GB – >6GB |

**Core trade-off:** CV models are cheap enough to run continuously in the background at high frame
rate but only classify the face. VLMs understand *why* (posture, surroundings) but are heavy
enough that they're better suited to selective triggering (e.g., only when the user speaks) than
continuous polling. Full detail in the survey doc above.

## Layout

| Folder | Contents |
|---|---|
| `ref/` | Source survey of the 14 candidate models — VRAM estimates and official repo links |
| `reports/` | Written analysis — currently the evaluation methodology; the results writeup will land here once benchmarking runs |
| `src/` | Benchmark runner scripts, once implemented (`src/cv/` for FER models, `src/vlm/` for VLMs, `src/eval/` for shared metrics helpers) |
| `data/` | Evaluation images/clips (FER2013 subset + self-collected stress-test set), once assembled |
| `results/` | Raw logs and aggregated metrics, once runs happen |

## Status

- [x] Survey 14 candidate models against the 6GB VRAM constraint
- [x] Define the two-track evaluation methodology (see `reports/evaluation_plan.md`)
- [ ] Assemble evaluation dataset (FER2013 subset + stress-test images/scenes)
- [ ] Implement benchmark runners and shared eval/logging helpers
- [ ] Run Track A (CV/FER) benchmarks
- [ ] Run Track B (VLM) benchmarks
- [ ] Publish final comparison + model recommendation

## References

### CV / FER models

- [Mini-Xception (`oarriaga/face_classification`)](https://github.com/oarriaga/face_classification) — CNN trained on FER2013, <100MB, CPU-viable
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — Google's CPU/mobile-optimized tracking + expression toolkit
- [`fer`](https://github.com/justinshenk/fer) — lightweight Python FER wrapper
- [DeepFace](https://github.com/serengil/deepface) — Python toolkit, multiple swappable backends (e.g. VGG-Face)
- [EmotiEffLib / HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition) — EfficientNet-based, PyTorch/ONNX
- [EfficientFace](https://github.com/zengqunzhao/EfficientFace) — ResNet-based FER
- [Py-Feat](https://github.com/cosanlab/py-feat) — facial action-unit + emotion toolkit
- [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) — facial action-unit tracking toolkit

### Vision-language models

- [Florence-2 (`microsoft/Florence-2-base`)](https://huggingface.co/microsoft/Florence-2-base) — smallest VLM candidate, 0.2B–0.7B
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — 3B VLM, fits budget at 4-bit quantization
- [Moondream2](https://github.com/vikhyat/moondream) — 1.8B compact VLM
- [PaliGemma (`google/paligemma-3b-pt-224`)](https://huggingface.co/google/paligemma-3b-pt-224) — 3B multimodal Gemma variant
- [MiniCPM-V 2.6](https://github.com/OpenBMB/MiniCPM-V) — 8B, only fits the 6GB budget at 4-bit quantization, near the ceiling
- [LLaVA-1.5-7B](https://github.com/haotian-liu/LLaVA) — documented as too heavy for the 6GB budget; kept in the survey as the upper-bound reference point
