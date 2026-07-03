# Visual Emotion Detection — Model Survey & Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository picks a model (or small pipeline) that reads a student's live emotional state
from webcam video during an LLM tutoring session. That state gets injected as context into the
tutoring LLM so it can steer its tone and pacing — the benchmark below exists to pick that model
responsibly, not as an end in itself. Full problem statement, constraints, and the recommended
production architecture: [`docs/problem_statement.md`](docs/problem_statement.md).

## Selection criteria

Every candidate is checked against these, in order; failing any one is a hard exclusion
regardless of the others, and the model goes straight to "Unlisted models" with no benchmark run:

1. **Capability** — must produce an emotion/affect signal directly, not an adjacent output
   (blendshapes, Action Units) with no label.
2. **License** — commercial-use compatible (MIT/Apache-2.0/etc.); research-only or
   non-commercial licenses are excluded regardless of accuracy.
3. **VRAM** — must fit inside the 6GB budget shared with the tutoring LLM itself, measured
   peak, not vendor-quoted.
4. **Latency** — realtime (≤300ms, synchronous), borderline (debounced async), or async-only
   (periodic enrichment, never gates a turn).
5. **Accuracy** — measured on the same test sample as every other model (see below), not a
   per-track metric that can't be compared across models.

Full reasoning per model: [`ref/visual_emotion_detection_models.md`](ref/visual_emotion_detection_models.md).

## Unified comparison

Every eligible model — CV/FER and VLM together — runs on the same 150-image sample drawn from
FER2013's test split, scored by the same accuracy metric. VLMs answer in one capped-token word,
mapped to the closest FER2013 label before scoring (`src/eval/label_mapping.py`) — never a long
reasoning prompt.

| Model | Type | Speed (median ms) | Accuracy |
|---|---|---|---|
| DeepFace | FER | 7.78 | 0.587 |
| HSEmotion / EmotiEffLib | FER | 8.63 | 0.533 |
| `fer` (mini-xception) | FER | 0.56 | 0.507 |
| PaliGemma-mix-224 (fast, one-word) | VLM | 225.06 | 0.500 |
| Moondream2 (fast, one-word) | VLM | 439.34 | 0.440 |

DeepFace leads on accuracy; `fer` is ~14x faster at a real but smaller accuracy cost. PaliGemma is
the strongest VLM here — 225ms median actually lands in the `realtime` fitness bucket (≤300ms),
though at 5.64GB peak VRAM it leaves almost no headroom for the LLM on a 6GB card. Both VLMs trail
DeepFace on accuracy — scene-reasoning strength doesn't help on tightly-cropped 48×48 face images,
and short-answer prompting caps how much either can express.

**Pending, not yet in the table:**
- **EfficientFace** — capable and license-clear, but its checkpoint arrives as a RAR archive and
  no extraction tool is installed in this environment. Its old (pre-unified) FER2013 number was
  0.525 accuracy / 8.96ms median — re-run on the unified sample once extracted.
- **EmotiEffLib engagement** — targets student engagement, not categorical emotion, so it isn't
  scored on the same FER2013 labels; see `ref/visual_emotion_detection_models.md` §5.

Full production-fitness table (license + real-time-bucket + shortlist):
[`reports/production_candidate_comparison.md`](reports/production_candidate_comparison.md).
Raw unified numbers: [`reports/unified_comparison.md`](reports/unified_comparison.md).

## Unlisted models

Fail capability, license, or VRAM-budget criteria — no accuracy/latency numbers exist for these
anywhere in this repo.

| Model | Reason |
|---|---|
| MediaPipe | No built-in emotion label, blendshapes/landmarks only |
| OpenFace 2.x | Action Units only, no emotion label; CMU non-commercial license |
| OpenFace 3.0 | Adds a real emotion head, but same CMU academic/non-commercial license as 2.x |
| Florence-2-base | Task-token model, not instruction-tuned — no open-ended emotion prompt exists |
| Qwen2.5-VL-3B-Instruct | Qwen Research License, non-commercial |
| LLaVA-1.5-7B | Exceeds the 6GB VRAM budget outright, no runner built |
| MiniCPM-V 2.6 | Cancelled — remote code incompatible with this repo's pinned `transformers==4.49.0`; ~16GB checkpoint deleted to free local disk space, not pursued further |

## Live evaluation

Informal single-session run against a real webcam (2026-07-03), not a controlled A/B — a
first pass to sanity-check responsiveness, not a benchmark.

**CV/FER models, 15s each, passive (no deliberate posing coordinated in advance):**

| Model | Frames w/ face | Dominant label | Label spread |
|---|---|---|---|
| `fer` | 372/372 | neutral (193) | happy 93, angry 39, surprise 29, sad 18 |
| DeepFace | 368/441 | sad (152) | angry 109, surprise 30, happy 33, fear 26, neutral 18 |
| HSEmotion | 451/451 | contempt (123) | neutral 112, surprise 100, happy 58, angry 58 |

Same face, same 15s window, three different dominant labels — DeepFace and HSEmotion in
particular disagree sharply (sad vs. contempt as the top label), consistent with the ~0.5
accuracy ceiling measured above; none of these should be trusted alone on a single frame.

**Moondream2 (fast, one-word), 20s:** caught a real transition — `neutral` (5 calls) →
`happy` (4 calls) → `Disgusted` (13 calls, held for the rest of the window). Whether that
last label reflects an actual sustained expression or the VLM latching onto something in the
scene is worth a deliberate follow-up run (see below).

**Follow-up needed:** this run wasn't coordinated with a specific pose sequence. A more useful
next pass: agree on a sequence of expressions to hold (e.g. neutral → happy → frown → neutral)
timed against the model output, to check whether the CV models and Moondream2-fast track actual
changes or just noise.

_Your verdict:_ _(add your own read on whether the above matches what you were actually doing
during the run)_

## Live webcam demo

```bash
.venv/Scripts/python src/cv/live_webcam_demo.py [seconds_per_model]
.venv/Scripts/python src/vlm/live_webcam_vlm_demo.py [seconds_per_model]
```

`live_webcam_demo.py` covers `fer`, DeepFace, HSEmotion, EfficientFace (per-frame, Haar-cascade
face crop). `live_webcam_vlm_demo.py` covers Moondream2, Qwen2.5-VL-3B, PaliGemma-mix-224,
MiniCPM-V 2.6, and Florence-2 (whole frame, one blocking call at a time — only one model loaded
on the GPU at once). Every script logs to `results/logs/` and writes a summary to `results/eval/`
on exit. Press `q` to quit early.

## Layout

| Folder | Contents |
|---|---|
| `ref/` | Model survey — selection criteria, capability/license verification, unlisted models |
| `reports/` | `unified_comparison.md` (the one comparable table), `production_candidate_comparison.md` (license + real-time-fitness + shortlist), `license_comparison.md`, `track_a_stress_test_results.md` (occlusion/lighting robustness) |
| `src/cv/` | CV/FER predictor factories (`live_webcam_demo.py`) and the occlusion/lighting stress-set tool |
| `src/vlm/` | VLM predictor factories (`live_webcam_vlm_demo.py`) |
| `src/eval/` | Shared harness: `model_registry.py` (every candidate + eligibility), `methods.py` (`unified_accuracy`, `latency_vram`, `session_fitness`), `run_method.py` (CLI driver), `build_unified_sample.py`, `label_mapping.py`, `licenses.py`, `aggregate_unified_comparison.py`, `aggregate_production_candidates.py` |
| `data/` | FER2013 manifests, `unified_eval/` (the shared 150-image sample), gitignored checkpoints/large images |
| `results/` | `eval/` — aggregated summaries; `logs/` — raw per-image/per-call outputs |

## Reproducing the results

```bash
py -3.12 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# Build the shared test sample, then run any model through it
.venv/Scripts/python src/eval/build_unified_sample.py
.venv/Scripts/python src/eval/run_method.py --model deepface --method unified_accuracy
.venv/Scripts/python src/eval/run_method.py --list   # all registered models/methods

# Regenerate both reports
.venv/Scripts/python src/eval/aggregate_unified_comparison.py
.venv/Scripts/python src/eval/aggregate_production_candidates.py
```

PaliGemma-mix-224 and MiniCPM-V 2.6 require `huggingface-cli login` (or `HF_TOKEN`) with an
account that has accepted both models' license terms on huggingface.co. EfficientFace needs its
upstream repo cloned to `data/models/efficientface_repo` and its RAF-DB checkpoint (linked in
that repo's README) extracted from its RAR archive into `data/models/efficientface/rafdb.pth`.

## Roadmap

- **This README** — up to date as of the unified comparison; update again once the two gated
  VLMs and EfficientFace are re-run (see "Pending, not yet in the table" above).
- **Production module** — not yet built. The comparison above picks the models; wiring them into
  an actual tutoring-loop signal (debounce, output contract, fallback state) is planned in
  [`docs/problem_statement.md`](docs/problem_statement.md) but not started.
- **Problem statement** — formalized in [`docs/problem_statement.md`](docs/problem_statement.md);
  update it if the target hardware, VRAM budget, or accuracy floor changes.
