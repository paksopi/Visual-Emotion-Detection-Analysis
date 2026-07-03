# Evaluation Plan — Visual Emotion Detection Models

Status: **planning only** — no models have been installed or run yet. This document defines how
the 14 candidates in [`ref/visual_emotion_detection_models.md`](../ref/visual_emotion_detection_models.md)
will be evaluated once benchmarking starts, so the eventual run is apples-to-apples instead of
ad hoc.

**Pre-check completed 2026-07-02:** before writing this plan, every candidate was verified against
its official docs to confirm it actually does emotion detection out of the box, not just an
adjacent vision task — see §4 of the survey doc. Two Track A models (**MediaPipe**, **OpenFace**)
don't output an emotion label at all (landmarks/AUs only) and are excluded from Track A's
head-to-head accuracy benchmark below unless a classifier is built on top of them first; they're
listed as face-tracking front ends, not candidates, until that exists. Two Track B models
(**Florence-2**, **PaliGemma**) needed a checkpoint correction (base → instruction-tuned) before
they're promptable at all — the survey doc has been corrected; Track B runs must use the corrected
checkpoints.

## 1. Why one eval protocol doesn't fit both categories

The two model families in the survey answer fundamentally different questions and can't be scored
on the same rubric:

- **CV/FER models** (Mini-Xception, `fer`, DeepFace, EmotiEffLib, EfficientFace, Py-Feat — 6 of the
  8 surveyed) output a fixed categorical label (Happy/Sad/Neutral/...) from a cropped face. This is
  a closed-set classification problem — standard ML metrics apply directly. MediaPipe and OpenFace
  are excluded from this head-to-head; they only produce landmarks/Action Units, not an emotion
  label (§4 of the survey doc).
- **VLMs** (Qwen2.5-VL-3B, Moondream2, MiniCPM-V, LLaVA, plus Florence-2 and PaliGemma once
  pointed at their instruction-tuned checkpoints) output free-text reasoning about the whole scene.
  There's no fixed label to score against; correctness has to be judged, not matched. Treating them
  with the same accuracy/F1 harness as the CV models would understate what they're actually for
  (contextual "why", not just "what").

So the plan splits into two tracks that share infrastructure (same hardware, same logging format)
but different scoring.

## 2. Track A — CV/FER models (quantitative)

**Dataset:** FER2013 test split (public, 7-class: angry, disgust, fear, happy, sad, surprise,
neutral) as the baseline benchmark, since every candidate in this track either was trained on it
(Mini-Xception) or supports it as an input format. Supplement with a small self-collected set
(~50–100 webcam frames) covering conditions FER2013 doesn't stress well: side lighting, partial
occlusion (glasses, hand near face), and off-axis head pose — these are the failure modes that
matter most for a live perception layer, not curated dataset accuracy.

**Metrics per model:**
| Metric | How |
|---|---|
| Accuracy / macro-F1 | Predicted label vs. FER2013 ground truth |
| Per-class confusion matrix | Where each model systematically confuses classes (e.g., fear↔surprise is a known FER2013 weak spot) |
| Inference latency | ms/frame, median + p95 over ≥200 frames, CPU and GPU separately where the model supports both |
| VRAM footprint | `torch.cuda.max_memory_allocated()` (or `nvidia-smi` polling for non-PyTorch backends) during a sustained run, not just model load |
| Face-detection overhead | Time spent in the upstream face-crop step (most of these need one) — charged separately since it's shared cost, not a property of the emotion model itself |
| Degradation under occlusion/lighting | Accuracy drop on the self-collected stress set vs. FER2013 baseline |

**Pass bar:** must run within the 6GB VRAM budget *alongside* whatever else the perception layer
needs concurrently — so the real target is a fraction of that budget (e.g., <1GB), not "fits in
6GB alone."

## 3. Track B — VLMs (qualitative + resource)

**Dataset:** a small fixed set of ~20–30 staged scene images/short clips, each with a known
"ground-truth" emotional context written by the tester beforehand (e.g., "person slumped at desk,
messy room, likely stressed/tired" vs. just "sad face"). Deliberately includes cases where the
face alone is ambiguous but scene context disambiguates it — that's the specific capability VLMs
are being evaluated for over the CV track.

**Scoring rubric (1–5 scale per image, scored blind to which model produced it):**
| Dimension | What it checks |
|---|---|
| Emotion correctness | Does the stated emotion match the intended ground truth? |
| Contextual grounding | Does the reasoning cite actual scene evidence (posture, environment, objects) rather than just the face? |
| Hallucination rate | Does it invent details not present in the image? (count, not scored 1–5) |
| Response usefulness | Would this reasoning actually help a downstream decision, or is it generic/hedgy? |

**Resource metrics:** same VRAM/latency measurement as Track A, but latency is expected to be
much higher (hundreds of ms to seconds) and is evaluated against a *selective-trigger* usage
pattern (only invoked when the user speaks, per the survey's trade-off note) rather than
per-frame real-time.

**Quantization note:** several candidates only fit the 6GB budget at 4-bit quantization
(Qwen2.5-VL-3B, PaliGemma, MiniCPM-V). Quantized accuracy/quality must be measured directly —
not assumed equal to the full-precision numbers reported upstream.

## 4. Shared test harness

Both tracks log to the same structure so results are comparable and reproducible:

```
data/            # FER2013 subset + self-collected stress-test images/scenes (gitignored if large; sample only committed)
src/cv/          # one runner script per Track A model, e.g. run_mini_xception.py
src/vlm/         # one runner script per Track B model, e.g. run_moondream2.py
src/eval/        # shared metrics: accuracy/F1/confusion matrix, latency timer, VRAM logger, rubric scoring sheet
results/logs/    # raw per-call timing + VRAM logs, one file per model per run
results/eval/    # aggregated metrics tables / confusion matrices / rubric scores
reports/         # this plan, plus the final comparison writeup once runs complete
```

Every model runner reports through the same `src/eval` logging helpers so the final comparison
table (accuracy or rubric score, latency, VRAM, license) is generated from consistent data rather
than hand-copied numbers.

## 5. Decision matrix (to fill in after runs)

Final model selection will weigh, in this order, matching the survey's stated hardware constraint:

1. **Fits the VRAM budget** alongside concurrent perception-layer load (hard filter, not scored)
2. **Accuracy/rubric score** on the relevant track
3. **Latency**, weighted by whether the use case is continuous (Track A) or event-triggered (Track B)
4. **License** for the intended deployment (commercial vs. research use)

This table is deliberately left empty until real runs produce numbers — no model is being
pre-selected here.

## 6. Explicitly out of scope for this pass

- Training or fine-tuning any model — evaluation is of off-the-shelf checkpoints only.
- Action-unit-level analysis (Py-Feat, OpenFace can do this) beyond what's needed to extract a
  coarse emotion label, since the perception layer's stated need is emotion category, not FACS
  coding.
- Multi-face scenes — single primary subject only, matching the perception layer's expected input.

## 7. Track C — production candidate comparison (2026-07-03)

Once Tracks A/B produced real numbers, a follow-on question emerged: picking exactly **one**
model for a real-time perception layer (detecting a student's emotional/engagement state during
a live LLM tutoring session, to steer tone/response) needs more than the raw per-track leaderboard
above — it needs latency judged against an actual real-time budget, and license eligibility
treated as a first-class, code-readable field rather than a separate manual cross-reference.

This is implemented as an **additive layer**, not a replacement for Tracks A/B — the runners,
aggregators (`aggregate_track_a.py`/`aggregate_track_b.py`), and their reports above are
untouched and still the source of truth for the FER2013/scene-context benchmark numbers.

**New pieces (`src/eval/`):**
- `model_registry.py` — a `ModelAdapter` per candidate, wrapping the *existing*
  `make_X_predictor()` closures already in `src/cv/live_webcam_demo.py` /
  `src/vlm/live_webcam_vlm_demo.py` (zero duplicated model-loading code), tagged with
  `output_kind`, `license_id`, and `production_eligible`.
- `methods.py` + `run_method.py` — a `Method` abstraction (`latency_vram`, `fer2013_accuracy`,
  `session_fitness`) so any registered model can be driven through any applicable method via one
  CLI: `python src/eval/run_method.py --model <key> --method <name> [--limit N]`.
- `session_fitness.py` — classifies a model's already-measured median latency into
  `realtime` (<300ms, safe synchronously in the chat-turn path), `borderline` (300ms-1s, viable
  async-with-debounce), or `async_only` (>1s, must run out-of-band). Not a new metric — a
  bucketing pass over numbers `LatencyTimer` already collected.
- `licenses.py` — `LICENSE_REGISTRY`, encoding `reports/license_comparison.md` as importable data.
- `aggregate_production_candidates.py` — scans every model's best summary + session-fitness
  verdict + license eligibility, writes `reports/production_candidate_comparison.md`.

**Two new candidates onboarded**, found while researching what's changed in the field since the
original 14-model survey (see `ref/visual_emotion_detection_models.md` §5 for full detail):
- **OpenFace 3.0** — checked and **rejected at the license gate** before any runner code was
  written: confirmed non-commercial-only (same CMU MultiComp Lab terms as OpenFace 2.x), despite
  genuinely adding a direct emotion head this repo's original survey found missing in 2.x.
- **EmotiEffLib engagement mode** (`emotiefflib` package, Apache-2.0, successor to the
  `hsemotion-onnx` already used here) — cleared and onboarded
  (`src/cv/run_emotiefflib_engagement.py`). Targets student engagement/session state directly,
  arguably closer to the actual production signal than 7/8-class categorical emotion.

Automated free-text rubric scoring (an LLM-as-judge replacement for the manual process in
`results/eval/track_b_rubric_scores.csv`) was scoped but **deferred** — manual scoring stays the
process for now.

Output: `reports/production_candidate_comparison.md`.
