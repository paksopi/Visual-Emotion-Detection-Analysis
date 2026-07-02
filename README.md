# Visual Emotion Detection — Model Survey & Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Survey, evaluation methodology, **and now real benchmark results** for
open-source visual emotion detection models, scoped for a local perception
layer under a strict **6GB VRAM** hardware budget. 14 candidate models were
surveyed across two families — lightweight facial-expression classifiers and
heavier vision-language models (VLMs) that reason about the whole scene — and
6 of them have now actually been run and measured on real hardware (RTX 3050
6GB Laptop GPU).

- Model survey (14 candidates, VRAM estimates, links): [`ref/visual_emotion_detection_models.md`](ref/visual_emotion_detection_models.md)
- Evaluation plan (how each family is measured): [`reports/evaluation_plan.md`](reports/evaluation_plan.md)
- **Results (real runs, this section is new): [`reports/model_comparison_results.md`](reports/model_comparison_results.md)**

## TL;DR results

**Track A (CV/FER, full FER2013 test split, 7,178 images):**

| Model | Accuracy | Macro-F1 | Median latency | Peak VRAM |
|---|---|---|---|---|
| **DeepFace** | **56.3%** | **0.547** | 8.1ms | 368MB |
| HSEmotion / EmotiEffLib | 52.7% | 0.499 | 8.5ms | 368MB |
| EfficientFace (RAF-DB) | 52.5% | 0.450 | 11.1ms | 17MB |
| `fer` (mini-xception) | 49.0% | 0.428 | **0.68ms** | 342MB |

DeepFace wins on accuracy; `fer`/mini-xception is ~10x faster but least
accurate. All four are trivially cheap on VRAM — latency, not memory, is
what actually differentiates them for a real-time perception layer.

**Track B (VLMs, 20-image scene-context set):**

| Model | Emotion correctness | Contextual grounding | Hallucinations | Median latency | Peak VRAM |
|---|---|---|---|---|---|
| Moondream2 (fp16) | 4.20/5 | **4.45/5** | **4** | **5.2s** | 4.49GB |
| Qwen2.5-VL-3B (4-bit) | 4.20/5 | 4.35/5 | 6 | 11.4s | **2.68GB** |

Both models are close on quality and both handle scene-disambiguation
correctly (e.g. reading an exertion-strained tennis face as effort, not
anger). The most useful finding is where they **fail identically**: both
missed a Santa hat and a reindeer-antlered dog in the same image, defaulting
to a generic read and inventing objects that aren't there — a shared
context-grounding gap, not a case that favors one model over the other.

Full numbers, confusion matrices, per-image rubric scores, and methodology
caveats: [`reports/model_comparison_results.md`](reports/model_comparison_results.md).

## Emotion-capability verification

Every candidate was checked against its official docs/model card to confirm
it actually produces an emotion output, not just an adjacent vision
capability. **10 of 14 confirmed capable out of the box; 4 needed a fix.**
Full detail and sources in §4 of the survey doc.

| Model | Verified? | Issue |
|---|---|---|
| Mini-Xception, `fer`, DeepFace, EmotiEffLib, Py-Feat | ✅ Yes | Direct emotion-label output, confirmed |
| EfficientFace | ✅ Yes, caveat | Direct output, but research code (no pip package) |
| **MediaPipe** | ⚠️ **No** | Only outputs blendshapes/landmarks — no built-in emotion label, needs a classifier bolted on top |
| **OpenFace** | ⚠️ **No** | Only outputs Action Units — no built-in emotion label, needs manual FACS→emotion mapping |
| Qwen2.5-VL-3B, Moondream2 | ✅ **Benchmarked** | See Track B results above |
| MiniCPM-V 2.6, LLaVA-1.5-7B | ✅ Yes, caveat | Genuinely promptable for emotion/affect reasoning, not yet benchmarked |
| **Florence-2** | ⚠️ **No — wrong checkpoint** | Linked `-base` is a task-token model, not instruction-tuned; can't take an open-ended emotion prompt without fine-tuning |
| **PaliGemma** | ⚠️ **No — wrong checkpoint (fixed)** | Linked `-pt-224` is raw pretrained, can't be prompted at all; survey now points to `-mix-224` instead |

## The two model families

| Family | What it outputs | Examples | VRAM range |
|---|---|---|---|
| CV / FER (lightweight) | Categorical label (Happy, Sad, Neutral, ...) from a cropped face | Mini-Xception, MediaPipe, `fer`, DeepFace, EmotiEffLib, EfficientFace, Py-Feat, OpenFace | <100MB – ~2GB |
| VLM (contextual) | Free-text reasoning about the whole scene (body language, environment) | Florence-2, Qwen2.5-VL-3B, Moondream2, PaliGemma, MiniCPM-V 2.6, LLaVA-1.5-7B | 0.9GB – >6GB |

**Core trade-off:** CV models are cheap enough to run continuously in the
background at high frame rate but only classify the face. VLMs understand
*why* (posture, surroundings) but are heavy enough that they're better suited
to selective triggering (e.g., only when the user speaks) than continuous
polling. The measured results above confirm this: Track A models run in
single-digit milliseconds at under 400MB, while Track B models take seconds
per call and gigabytes of VRAM — the plan's predicted trade-off held up in
practice.

## Layout

| Folder | Contents |
|---|---|
| `ref/` | Source survey of the 14 candidate models — VRAM estimates and official repo links |
| `reports/` | `evaluation_plan.md` (methodology) and `model_comparison_results.md` (actual results + caveats) |
| `src/cv/` | Track A runner scripts — one per CV/FER model (`run_deepface.py`, `run_fer.py`, `run_hsemotion.py`, `run_efficientface.py`) |
| `src/vlm/` | Track B runner scripts — one per VLM (`run_moondream2.py`, `run_qwen25vl.py`) |
| `src/eval/` | Shared harness: latency timer, VRAM tracker, classification metrics, rubric scoring, run logging, and the `aggregate_track_a.py` / `aggregate_track_b.py` comparison-table generators |
| `data/` | FER2013 manifests + a 20-image Track B scene-context set with authored ground truth (`data/track_b/README.md` explains provenance/limitations); large downloaded images and model checkpoints are gitignored and regenerate via the runner scripts |
| `results/` | `eval/` — aggregated comparison tables, confusion matrices, and rubric scores; `logs/` — raw per-image/per-call outputs for every run |

## Status

- [x] Survey 14 candidate models against the 6GB VRAM constraint
- [x] Define the two-track evaluation methodology (see `reports/evaluation_plan.md`)
- [x] Verify each model actually does emotion detection, not just an adjacent vision task
- [x] Assemble evaluation dataset — full FER2013 test split (Track A) + 20-image scene-context set (Track B)
- [x] Implement benchmark runners and shared eval/logging helpers
- [x] Run Track A (CV/FER) benchmarks — 4 of 6 in-scope candidates (`fer`/mini-xception, DeepFace, HSEmotion, EfficientFace)
- [x] Run Track B (VLM) benchmarks — 2 of 6 candidates (Moondream2, Qwen2.5-VL-3B 4-bit)
- [x] Publish results + decision-matrix writeup (`reports/model_comparison_results.md`)
- [ ] Track A self-collected occlusion/lighting stress set (needs a physical camera)
- [ ] Track B staged photos + human-authored ground truth (current set is an AI-authored stand-in, see `data/track_b/README.md`)
- [ ] Py-Feat (blocked — see results doc), Florence-2, PaliGemma-mix, MiniCPM-V 2.6, LLaVA-1.5-7B
- [ ] License comparison across the shortlisted models

## References

### CV / FER models

- [Mini-Xception (`oarriaga/face_classification`)](https://github.com/oarriaga/face_classification) — CNN trained on FER2013, <100MB, CPU-viable, direct 7-class emotion output ✅ — **benchmarked via the `fer` library, which ships these weights internally**
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — Google's CPU/mobile-optimized tracking toolkit; ⚠️ blendshapes/landmarks only, no built-in emotion label
- [`fer`](https://github.com/justinshenk/fer) — lightweight Python FER wrapper, direct emotion dict output ✅ **benchmarked: 49.0% acc, 0.68ms median**
- [DeepFace](https://github.com/serengil/deepface) — Python toolkit, multiple swappable backends (e.g. VGG-Face), direct `dominant_emotion` output ✅ **benchmarked: 56.3% acc, 8.1ms median — best accuracy**
- [EmotiEffLib / HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition) — EfficientNet-based, PyTorch/ONNX, direct emotion label ✅ **benchmarked: 52.7% acc, 8.5ms median**
- [EfficientFace](https://github.com/zengqunzhao/EfficientFace) — ResNet-based FER, direct output but research code (no pip package) ✅⚠️ **benchmarked: 52.5% acc, 11.1ms median — required empirically recovering the RAF-DB label order, since upstream never documents it (see results doc)**
- [Py-Feat](https://github.com/cosanlab/py-feat) — facial action-unit toolkit with a dedicated emotion output field, not AUs alone ✅ — **attempted, blocked by an FFmpeg/torchcodec dependency issue on this machine (see results doc)**
- [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) — facial action-unit tracking toolkit; ⚠️ Action Units only, no built-in emotion label

### Vision-language models

- [Florence-2 (`microsoft/Florence-2-base`)](https://huggingface.co/microsoft/Florence-2-base) — smallest VLM candidate, 0.2B–0.7B; ⚠️ task-token base model, not instruction-tuned, can't take open-ended emotion prompts as-is
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — 3B VLM, fits budget at 4-bit quantization, promptable for emotion reasoning ✅ **benchmarked: 4.20/5 emotion correctness, 2.68GB VRAM, 11.4s median latency**
- [Moondream2](https://github.com/vikhyat/moondream) — 1.8B compact VLM, promptable for emotion reasoning ✅ **benchmarked: 4.20/5 emotion correctness, 4.49GB VRAM, 5.2s median latency — best grounding/hallucination rate**
- [PaliGemma (`google/paligemma-3b-mix-224`)](https://huggingface.co/google/paligemma-3b-mix-224) — 3B multimodal Gemma variant; **survey corrected from the raw `-pt-224` pretrained checkpoint**, which cannot be prompted at all, to the instruction-tuned `-mix-224`
- [MiniCPM-V 2.6](https://github.com/OpenBMB/MiniCPM-V) — 8B, only fits the 6GB budget at 4-bit quantization, near the ceiling, promptable for emotion reasoning ✅⚠️ not yet benchmarked
- [LLaVA-1.5-7B](https://github.com/haotian-liu/LLaVA) — documented as too heavy for the 6GB budget; kept in the survey as the upper-bound reference point, promptable for emotion reasoning ✅⚠️ not yet benchmarked

## Reproducing the results

```bash
py -3.12 -m venv .venv
.venv/Scripts/pip install -r <deps used per runner — see src/cv/*.py and src/vlm/*.py docstrings>

# Track A (writes results/eval/ + results/logs/)
.venv/Scripts/python src/cv/run_deepface.py
.venv/Scripts/python src/cv/run_fer.py
.venv/Scripts/python src/cv/run_hsemotion.py
.venv/Scripts/python src/cv/run_efficientface.py
.venv/Scripts/python src/eval/aggregate_track_a.py

# Track B
.venv/Scripts/python src/vlm/run_moondream2.py
.venv/Scripts/python src/vlm/run_qwen25vl.py
.venv/Scripts/python src/eval/aggregate_track_b.py
```

FER2013 test images and downloaded model checkpoints are gitignored (large,
regeneratable) — see the runner scripts' docstrings for where they're pulled
from (`clip-benchmark/wds_fer2013` on Hugging Face for FER2013; Hugging Face
Hub / the EfficientFace author's Google Drive links for model weights).
