# Visual Emotion Detection — Model Survey & Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Survey, evaluation methodology, **and now real benchmark results** for
open-source visual emotion detection models, scoped for a local perception
layer under a strict **6GB VRAM** hardware budget. 14 candidate models were
surveyed across two families — lightweight facial-expression classifiers and
heavier vision-language models (VLMs) that reason about the whole scene — and
9 of them have now actually been run and measured on real hardware (RTX 3050
6GB Laptop GPU): 7 on emotion accuracy/rubric, plus MediaPipe and Florence-2
on their native (non-emotion) capability.

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
| Py-Feat (Detectorv1) | 48.9% | 0.427 | 101.7ms | 932MB |

DeepFace wins on accuracy; `fer`/mini-xception is ~10x faster but least
accurate. All five are trivially cheap on VRAM — latency, not memory, is
what actually differentiates them for a real-time perception layer.

**Native-capability runs (no emotion output, run on what each model actually
does instead):**

| Model | What it ran | Result | Median latency | Peak VRAM |
|---|---|---|---|---|
| MediaPipe FaceLandmarker | Face detection + 52 blendshape coefficients | 85.0% face-detection rate on FER2013; `mouthSmile*` dominates for `happy`, etc. | 8.8ms | n/a (CPU) |
| Florence-2-base | `<DETAILED_CAPTION>` dense scene captioning | Usable as a cheap scene-description signal; can't take an open-ended emotion prompt at all | 672.7ms | 2.05GB |

Neither has a built-in emotion label (confirmed in the capability table
below), so instead of forcing an "N/A" into the accuracy tables, each was
benchmarked on the capability it actually has. Full numbers:
`results/eval/track_a_comparison.md` / `track_b_comparison.md`
§"Native-capability results".

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

## Live webcam demo

Every model in this repo can now be pointed at your own webcam, split across
three scripts (different venvs / capture patterns):

| Script | Venv | Models | Pattern |
|---|---|---|---|
| `src/cv/live_webcam_demo.py` | `.venv/` | `fer`, DeepFace, HSEmotion, EfficientFace, MediaPipe | per-frame, Haar-cascade face crop |
| `src/cv/live_webcam_pyfeat_demo.py` | `.venv-pyfeat/` | Py-Feat | per-frame, Haar-cascade face crop (fed as a tensor — Py-Feat's file-based `detect()` mode can't take a live frame) |
| `src/vlm/live_webcam_vlm_demo.py` | `.venv/` | Moondream2, Qwen2.5-VL-3B (4-bit), Florence-2 | whole frame, one blocking call at a time (seconds per call) — only one model loaded on the GPU at once |

```bash
.venv/Scripts/python src/cv/live_webcam_demo.py [seconds_per_model]
.venv-pyfeat/Scripts/python src/cv/live_webcam_pyfeat_demo.py [seconds]
.venv/Scripts/python src/vlm/live_webcam_vlm_demo.py [seconds_per_model]
```

MediaPipe and Florence-2 have no emotion label (see above), so their live
"prediction" is a stand-in: MediaPipe shows its single most active
blendshape, Florence-2 runs its native `<DETAILED_CAPTION>` task instead of
the emotion prompt. Every script logs per-frame/per-call predictions to
`results/logs/` and writes a summary to `results/eval/` on exit, same as the
batch runs. Press `q` at any time to quit early.

A 15s-per-model smoke test against a
live face (single subject, sequential — not simultaneous — slices) showed:

- **HSEmotion** was the most reliable in practice: 100% face-detection rate
  (0 dropped frames) and a balanced label spread, consistent with its
  second-best FER2013 accuracy above.
- **EfficientFace** showed a strong `surprise` bias (66% of frames) live,
  consistent with the label-order calibration being empirically inferred
  rather than confirmed (see `src/cv/run_efficientface.py` docstring).
- `fer` and DeepFace both skewed toward `happy`/`neutral` with no `fear` or
  `disgust` predictions in the sample window.

This is a single anecdotal run, not a benchmark — useful as a sanity check
and quick qualitative feel for each model, not a replacement for the FER2013
numbers above.

## Emotion-capability verification

Every candidate was checked against its official docs/model card to confirm
it actually produces an emotion output, not just an adjacent vision
capability. **10 of 14 confirmed capable out of the box; 4 needed a fix.**
Full detail and sources in §4 of the survey doc.

| Model | Verified? | Issue |
|---|---|---|
| Mini-Xception, `fer`, DeepFace, EmotiEffLib, Py-Feat | ✅ Yes | Direct emotion-label output, confirmed — **all benchmarked above** |
| EfficientFace | ✅ Yes, caveat | Direct output, but research code (no pip package) |
| **MediaPipe** | ⚠️ **No** | Only outputs blendshapes/landmarks — no built-in emotion label; **run on its native capability instead, see above** |
| **OpenFace** | ⚠️ **No** | Only outputs Action Units — no built-in emotion label, needs manual FACS→emotion mapping |
| Qwen2.5-VL-3B, Moondream2 | ✅ **Benchmarked** | See Track B results above |
| MiniCPM-V 2.6, LLaVA-1.5-7B | ✅ Yes, caveat | Genuinely promptable for emotion/affect reasoning, not yet benchmarked |
| **Florence-2** | ⚠️ **No — wrong checkpoint** | Linked `-base` is a task-token model, not instruction-tuned; can't take an open-ended emotion prompt without fine-tuning; **run on its native captioning task instead, see above** |
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
| `reports/` | `evaluation_plan.md` (methodology), `model_comparison_results.md` (actual results + caveats), `license_comparison.md` (license per shortlisted model), `track_a_stress_test_results.md` (occlusion/lighting robustness check) |
| `src/cv/` | Track A runner scripts — one per CV/FER model (`run_deepface.py`, `run_fer.py`, `run_hsemotion.py`, `run_efficientface.py`, `run_pyfeat.py`), plus `run_mediapipe.py` (native blendshapes/landmarks, no emotion label) — and `live_webcam_demo.py` / `live_webcam_pyfeat_demo.py` (live-camera demos, see below) and `collect_track_a_stress_set.py` (occlusion/lighting stress-set capture + eval) |
| `src/vlm/` | Track B runner scripts — one per VLM (`run_moondream2.py`, `run_qwen25vl.py`), plus `run_florence2.py` (native dense captioning, no open-ended emotion prompting) and `live_webcam_vlm_demo.py` (live-camera demo, see below) |
| `src/eval/` | Shared harness: latency timer, VRAM tracker, classification metrics, rubric scoring, run logging, and the `aggregate_track_a.py` / `aggregate_track_b.py` comparison-table generators, plus Track C's `model_registry.py` / `methods.py` / `run_method.py` / `session_fitness.py` / `licenses.py` / `aggregate_production_candidates.py` (see below) |
| `data/` | FER2013 manifests + a 20-image Track B scene-context set with authored ground truth (`data/track_b/README.md` explains provenance/limitations); large downloaded images and model checkpoints are gitignored and regenerate via the runner scripts; `track_a_stress/` (self-collected webcam photos) is gitignored — personal face images, not published |
| `results/` | `eval/` — aggregated comparison tables, confusion matrices, and rubric scores; `logs/` — raw per-image/per-call outputs for every run |

## Track C — picking one production model

Tracks A/B above answer "how does each model perform." Track C answers a
narrower, practical question on top: **which single model should actually
ship**, for a real-time perception layer that reads a student's
emotional/engagement state during a live LLM tutoring session and steers the
LLM's tone/response. That needs latency judged against an actual real-time
budget (not just "fast" in the abstract) and license eligibility as a
first-class, checkable field — not just performance ranking.

It's an **additive layer** — every Track A/B runner, aggregator, and report
above is untouched and still authoritative for the raw benchmark numbers.

```bash
# Run any registered model through any applicable method
.venv/Scripts/python src/eval/run_method.py --model deepface --method latency_vram
.venv/Scripts/python src/eval/run_method.py --model deepface --method fer2013_accuracy
.venv/Scripts/python src/eval/run_method.py --model deepface --method session_fitness
.venv/Scripts/python src/eval/run_method.py --list   # show all registered models/methods

# Regenerate the full comparison table + shortlist
.venv/Scripts/python src/eval/session_fitness.py               # batch latency-bucket pass
.venv/Scripts/python src/eval/aggregate_production_candidates.py
```

Output: [`reports/production_candidate_comparison.md`](reports/production_candidate_comparison.md)
— one table across every registered model (latency/VRAM, accuracy or rubric
score, real-time-fitness bucket, license/commercial-use eligibility), sorted
so eligible/fast/accurate candidates surface first, closing with an
auto-generated shortlist. Two new candidates were checked while building
this: **OpenFace 3.0** (rejected — confirmed non-commercial-only license,
see [`ref/visual_emotion_detection_models.md`](ref/visual_emotion_detection_models.md)
§5) and **EmotiEffLib engagement mode** (cleared — Apache-2.0, targets
student engagement directly, `src/cv/run_emotiefflib_engagement.py`).

**VLM latency is decode-bound, not vision-bound** — cost scales with how
many tokens the model *generates*, not the image itself. Forcing a one-word
answer (`moondream2_fast`, `qwen25vl3b_4bit_fast` in the registry) instead of
a reasoned explanation cuts Moondream2 from ~5s to **~0.86s/call** (crosses
into the `borderline` session-fitness bucket) and Qwen2.5-VL-3B from
~10-17s to **~1.2-1.4s/call** — near Florence-2's native-captioning speed,
but with a real emotion label Florence-2 fundamentally cannot produce (it's
a task-token model, not instruction-tuned — there's no prompt that adds
emotion detection to it short of fine-tuning a new task token onto it with
labeled data). Trade-off: no scene-grounded reasoning, just the label.

## Status

Survey, methodology, dataset, runners, and benchmarks are all complete — 5 of
6 in-scope Track A emotion-accuracy candidates ran (`fer`/mini-xception,
DeepFace, HSEmotion, EfficientFace, Py-Feat), plus MediaPipe on its native
capability (no emotion label). 2 of 6 Track B candidates ran on the emotion
rubric (Moondream2, Qwen2.5-VL-3B), plus Florence-2 on its native capability.
Remaining open items:

- **OpenFace** — no built-in emotion label; would need a manual FACS→emotion
  mapping, not attempted.
- **Track B staged photos + human-authored ground truth** — current set is an
  AI-authored stand-in (see `data/track_b/README.md`); needs a human to
  pose/label photos, not automatable.
- **PaliGemma-mix, MiniCPM-V 2.6, LLaVA-1.5-7B** — not attempted; each needs a
  multi-GB checkpoint download, and MiniCPM-V 2.6/LLaVA-1.5-7B are already
  documented above as at-or-over this repo's 6GB VRAM budget.

## References

### CV / FER models

- [Mini-Xception (`oarriaga/face_classification`)](https://github.com/oarriaga/face_classification) — CNN trained on FER2013, <100MB, CPU-viable, direct 7-class emotion output ✅ — **benchmarked via the `fer` library, which ships these weights internally**
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — Google's CPU/mobile-optimized tracking toolkit; ⚠️ blendshapes/landmarks only, no built-in emotion label ✅ **native capability benchmarked: 85.0% face-detection rate, 8.83ms median latency**
- [`fer`](https://github.com/justinshenk/fer) — lightweight Python FER wrapper, direct emotion dict output ✅ **benchmarked: 49.0% acc, 0.68ms median**
- [DeepFace](https://github.com/serengil/deepface) — Python toolkit, multiple swappable backends (e.g. VGG-Face), direct `dominant_emotion` output ✅ **benchmarked: 56.3% acc, 8.1ms median — best accuracy**
- [EmotiEffLib / HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition) — EfficientNet-based, PyTorch/ONNX, direct emotion label ✅ **benchmarked: 52.7% acc, 8.5ms median**
- [EfficientFace](https://github.com/zengqunzhao/EfficientFace) — ResNet-based FER, direct output but research code (no pip package) ✅⚠️ **benchmarked: 52.5% acc, 11.1ms median — required empirically recovering the RAF-DB label order, since upstream never documents it (see results doc)**
- [Py-Feat](https://github.com/cosanlab/py-feat) — facial action-unit toolkit with a dedicated emotion output field, not AUs alone ✅ **benchmarked: 48.9% acc, 0.427 macro-F1, 101.7ms median (needed its own venv — its `torchcodec` dependency conflicts with this repo's shared `torch==2.6.0` pin, see results doc)**
- [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) — facial action-unit tracking toolkit; ⚠️ Action Units only, no built-in emotion label

### Vision-language models

- [Florence-2 (`microsoft/Florence-2-base`)](https://huggingface.co/microsoft/Florence-2-base) — smallest VLM candidate, 0.2B–0.7B; ⚠️ task-token base model, not instruction-tuned, can't take open-ended emotion prompts as-is ✅ **native captioning task benchmarked: 672.7ms median, 2.05GB VRAM**
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — 3B VLM, fits budget at 4-bit quantization, promptable for emotion reasoning ✅ **benchmarked: 4.20/5 emotion correctness, 2.68GB VRAM, 11.4s median latency**
- [Moondream2](https://github.com/vikhyat/moondream) — 1.8B compact VLM, promptable for emotion reasoning ✅ **benchmarked: 4.20/5 emotion correctness, 4.49GB VRAM, 5.2s median latency — best grounding/hallucination rate**
- [PaliGemma (`google/paligemma-3b-mix-224`)](https://huggingface.co/google/paligemma-3b-mix-224) — 3B multimodal Gemma variant; **survey corrected from the raw `-pt-224` pretrained checkpoint**, which cannot be prompted at all, to the instruction-tuned `-mix-224`
- [MiniCPM-V 2.6](https://github.com/OpenBMB/MiniCPM-V) — 8B, only fits the 6GB budget at 4-bit quantization, near the ceiling, promptable for emotion reasoning ✅⚠️ not yet benchmarked
- [LLaVA-1.5-7B](https://github.com/haotian-liu/LLaVA) — documented as too heavy for the 6GB budget; kept in the survey as the upper-bound reference point, promptable for emotion reasoning ✅⚠️ not yet benchmarked

## Reproducing the results

```bash
py -3.12 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
# torch needs its CUDA index explicitly (plain pip resolves a CPU-only build - see requirements.txt header):
.venv/Scripts/pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# Track A (writes results/eval/ + results/logs/)
.venv/Scripts/python src/cv/run_deepface.py
.venv/Scripts/python src/cv/run_fer.py
.venv/Scripts/python src/cv/run_hsemotion.py
.venv/Scripts/python src/cv/run_efficientface.py
.venv/Scripts/python src/cv/run_mediapipe.py     # native capability, no emotion label
.venv/Scripts/python src/eval/aggregate_track_a.py

# Py-Feat needs its own venv (torchcodec/torch version conflict, see results doc)
py -3.12 -m venv .venv-pyfeat
.venv-pyfeat/Scripts/pip install -r requirements-pyfeat.txt
.venv-pyfeat/Scripts/python src/cv/run_pyfeat.py

# Track B
.venv/Scripts/python src/vlm/run_moondream2.py
.venv/Scripts/python src/vlm/run_qwen25vl.py
.venv/Scripts/python src/vlm/run_florence2.py     # native capability, no open-ended emotion prompt
.venv/Scripts/python src/eval/aggregate_track_b.py
```

FER2013 test images and downloaded model checkpoints are gitignored (large,
regeneratable) — see the runner scripts' docstrings for where they're pulled
from (`clip-benchmark/wds_fer2013` on Hugging Face for FER2013; Hugging Face
Hub / the EfficientFace author's Google Drive links for model weights).
