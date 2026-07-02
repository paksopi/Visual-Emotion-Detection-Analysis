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

## Emotion-capability verification

Every candidate was re-checked against its official docs/model card to confirm it actually
produces an emotion output, not just an adjacent vision capability. **10 of 14 confirmed capable
out of the box; 4 needed a fix.** Full detail and sources in §4 of the survey doc.

| Model | Verified? | Issue |
|---|---|---|
| Mini-Xception, `fer`, DeepFace, EmotiEffLib, Py-Feat | ✅ Yes | Direct emotion-label output, confirmed |
| EfficientFace | ✅ Yes, caveat | Direct output, but research code (no pip package) |
| **MediaPipe** | ⚠️ **No** | Only outputs blendshapes/landmarks — no built-in emotion label, needs a classifier bolted on top |
| **OpenFace** | ⚠️ **No** | Only outputs Action Units — no built-in emotion label, needs manual FACS→emotion mapping |
| Qwen2.5-VL-3B, Moondream2, MiniCPM-V 2.6, LLaVA-1.5-7B | ✅ Yes, caveat | Genuinely promptable for emotion/affect reasoning, but not yet measured — see evaluation plan |
| **Florence-2** | ⚠️ **No — wrong checkpoint** | Linked `-base` is a task-token model, not instruction-tuned; can't take an open-ended emotion prompt without fine-tuning |
| **PaliGemma** | ⚠️ **No — wrong checkpoint (fixed)** | Linked `-pt-224` is raw pretrained, can't be prompted at all; survey now points to `-mix-224` instead |

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
- [x] Verify each model actually does emotion detection, not just an adjacent vision task (10/14 confirmed, 4 fixed/flagged — see §4 of the survey doc)
- [ ] Assemble evaluation dataset (FER2013 subset + stress-test images/scenes)
- [ ] Implement benchmark runners and shared eval/logging helpers
- [ ] Run Track A (CV/FER) benchmarks
- [ ] Run Track B (VLM) benchmarks
- [ ] Publish final comparison + model recommendation

## References

### CV / FER models

- [Mini-Xception (`oarriaga/face_classification`)](https://github.com/oarriaga/face_classification) — CNN trained on FER2013, <100MB, CPU-viable, direct 7-class emotion output ✅
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — Google's CPU/mobile-optimized tracking toolkit; ⚠️ blendshapes/landmarks only, no built-in emotion label
- [`fer`](https://github.com/justinshenk/fer) — lightweight Python FER wrapper, direct emotion dict output ✅
- [DeepFace](https://github.com/serengil/deepface) — Python toolkit, multiple swappable backends (e.g. VGG-Face), direct `dominant_emotion` output ✅
- [EmotiEffLib / HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition) — EfficientNet-based, PyTorch/ONNX, direct emotion label ✅
- [EfficientFace](https://github.com/zengqunzhao/EfficientFace) — ResNet-based FER, direct output but research code (no pip package) ✅⚠️
- [Py-Feat](https://github.com/cosanlab/py-feat) — facial action-unit toolkit with a dedicated emotion output field, not AUs alone ✅
- [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) — facial action-unit tracking toolkit; ⚠️ Action Units only, no built-in emotion label

### Vision-language models

- [Florence-2 (`microsoft/Florence-2-base`)](https://huggingface.co/microsoft/Florence-2-base) — smallest VLM candidate, 0.2B–0.7B; ⚠️ task-token base model, not instruction-tuned, can't take open-ended emotion prompts as-is
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — 3B VLM, fits budget at 4-bit quantization, promptable for emotion reasoning ✅⚠️
- [Moondream2](https://github.com/vikhyat/moondream) — 1.8B compact VLM, promptable for emotion reasoning ✅⚠️
- [PaliGemma (`google/paligemma-3b-mix-224`)](https://huggingface.co/google/paligemma-3b-mix-224) — 3B multimodal Gemma variant; **survey corrected from the raw `-pt-224` pretrained checkpoint**, which cannot be prompted at all, to the instruction-tuned `-mix-224`
- [MiniCPM-V 2.6](https://github.com/OpenBMB/MiniCPM-V) — 8B, only fits the 6GB budget at 4-bit quantization, near the ceiling, promptable for emotion reasoning ✅⚠️
- [LLaVA-1.5-7B](https://github.com/haotian-liu/LLaVA) — documented as too heavy for the 6GB budget; kept in the survey as the upper-bound reference point, promptable for emotion reasoning ✅⚠️
