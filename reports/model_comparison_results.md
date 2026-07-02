# Visual Emotion Detection — Evaluation Results

Status: **benchmarks complete** (including Py-Feat and the MediaPipe/Florence-2
native-capability runs added after the first pass), following the protocol in
[`evaluation_plan.md`](evaluation_plan.md). All numbers below come from actual
runs on this machine (RTX 3050 6GB Laptop GPU, Python 3.12 venv) — see
`results/eval/*_summary.json` and `results/logs/*.jsonl` for raw data, and
`src/cv/`, `src/vlm/`, `src/eval/` for the runner/harness code.

## 0. What's real vs. substituted here

- **Track A** ran on the actual FER2013 test split (7,178 images, pulled from
  the `clip-benchmark/wds_fer2013` HF mirror) — this is the real benchmark the
  plan called for.
- Track A's **self-collected stress set** (webcam frames under occlusion/side
  lighting) has since been produced — see
  [`track_a_stress_test_results.md`](track_a_stress_test_results.md).
  FER2013 itself is still pre-cropped 48×48 faces with no scene for a
  detector to search, so face-detection overhead isn't measured on that split
  (see `src/cv/run_fer.py` docstring).
- **Track B**'s image set is a substitute: 20 real photos sourced from the
  EMOTIC/MS-COCO image corpus (not staged), with ground truth **authored by
  the AI assistant that ran this evaluation**, not a human tester. See
  `data/track_b/README.md`. Treat Track B's numbers as illustrative of
  methodology, not a validated benchmark.
- OpenFace-as-classifier was not run (no built-in emotion label, would need a
  manual FACS→emotion mapping — see survey doc).
- MediaPipe and Florence-2 have no built-in emotion output, so instead of an
  emotion-accuracy score they were run on their actual native capability
  (blendshapes/landmarks, dense captioning) — see §1 and §2.

## 1. Track A — CV/FER models (quantitative)

5 of the 6 in-scope candidates ran (OpenFace excluded per plan §4 — no
built-in emotion label; MediaPipe likewise has none, but was run on its
native capability instead, see below). Py-Feat and a from-scratch
EfficientFace attempt were both "best-effort" per the plan; both ultimately
succeeded.

**Py-Feat: ran, after a dependency fix.** It initially failed to import
because its video-decoding dependency (`torchcodec`) requires `torch>=2.11`,
which conflicts with this repo's `torch==2.6.0` pin used by every other Track
A model — not really an FFmpeg problem, though a shared-library FFmpeg build
(`bin/ffmpeg-shared/`, gitignored) is also required on Windows since the
system FFmpeg here is static-only. Fixed by giving Py-Feat its own venv
(`.venv-pyfeat/`) instead of forcing it into the shared one. Result: 48.9%
accuracy, 0.427 macro-F1, 101.7ms median latency, 932MB peak VRAM — direct
7-class output (no extra classes, unlike HSEmotion's 8-class output).

**MediaPipe: no emotion label, run on its native task instead.**
FaceLandmarker outputs face landmarks + 52 ARKit-style blendshape
coefficients, not an emotion label — forcing it into the accuracy table
would just be "N/A" everywhere. Ran instead on face-detection rate and
per-frame blendshape activity: 85.0% face-detection rate on FER2013 (lower
than the FER-family models since FER2013's 48×48 crops are harder for a
general-purpose landmarker), 8.83ms median latency, and per-label top-5
blendshape means (e.g. `mouthSmileLeft`/`mouthSmileRight` dominate for
`happy`) as a proxy for whether its landmark geometry tracks
emotion-adjacent movement at all. Full breakdown:
`results/eval/track_a_comparison.md` §"Native-capability results".

**EfficientFace: ran, with a label-order correction worth flagging.** The
upstream repo (manual checkpoint via Google Drive, no pip package) trains on
RAF-DB via `ImageFolder` but never documents the resulting class-index order
in either `main.py` or its own test script. The commonly-cited RAF-DB numeric
convention was tried first and **did not match** this checkpoint's actual
output (near-0% accuracy). The real mapping was recovered empirically:
raw-index predictions vs. FER2013 ground truth on a 1000-image calibration
slice, solved via Hungarian assignment for the best label permutation
(51.6% agreement — high enough to trust it's a real signal, not noise). Final
numbers below are on the held-out remaining 6,178 images, not the calibration
slice.

| Model | N | Accuracy | Macro-F1 | Median latency (ms) | p95 latency (ms) | Peak VRAM/RAM (MB) |
|---|---|---|---|---|---|---|
| DeepFace | 7178 | **0.563** | 0.547 | 8.12 | 10.76 | 368 |
| HSEmotion / EmotiEffLib | 7178 | 0.527 | 0.499 | 8.54 | 10.85 | 368 |
| EfficientFace (RAF-DB checkpoint) | 6178 | 0.525 | 0.450 | 11.07 | 15.51 | 17 |
| fer (justinshenk/fer, mini-xception weights) | 7178 | 0.490 | 0.428 | **0.68** | **0.97** | 342 |
| Py-Feat (Detectorv1) | 7178 | 0.489 | 0.427 | 101.70 | 109.98 | 932 |

Full confusion matrices: `results/eval/track_a_comparison.md`.

**Reading the results:**
- **fer / Mini-Xception** — the survey listed these as two separate
  candidates, but the `fer` library ships oarriaga's original FER2013
  mini-xception weights internally, so one run covers both. It's by far the
  fastest (sub-millisecond) but also the least accurate — consistent with it
  being the oldest, smallest architecture of the four.
- **DeepFace** wins on accuracy with mid-pack latency — best overall pick if
  accuracy is the priority.
- **HSEmotion** has an 8th class ("contempt") that FER2013 has no ground
  truth for, so any contempt prediction is scored as a miss — a real
  train/eval label-mismatch this model would hit in production against
  FER2013-style labels, not a harness artifact.
- All five models struggle most on **fear**, frequently confusing it with
  sad or neutral (see confusion matrices) — consistent with fear being
  FER2013's well-known weak class.
- **VRAM is mostly a non-issue**: four of five sit under 400MB; Py-Feat is
  the outlier at 932MB (still comfortably inside the 6GB budget). Latency is
  where they actually differentiate — sub-15ms for four of them, but Py-Feat
  is ~10-100x slower at 101.7ms median, the tradeoff for its dedicated
  action-unit pipeline.

## 2. Track B — VLMs (qualitative + resource)

Both in-budget, non-quantization-required-at-fp16 candidates ran: Moondream2
(fp16) and Qwen2.5-VL-3B-Instruct (4-bit nf4, per the plan's quantization
note). PaliGemma, MiniCPM-V, and LLaVA were not attempted this pass
(time/scope).

**Florence-2: no open-ended emotion prompting, run on its native task
instead.** Florence-2-base is a task-token model, not instruction-tuned —
`<VQA>What emotion is the person feeling?` decodes to garbage
(`QA>Emotion`), not a real answer, and there's no larger/instruction-tuned
Florence-2 checkpoint to swap in (unlike PaliGemma's `-pt` → `-mix` fix). Ran
instead on its actual native task, `<DETAILED_CAPTION>` dense scene
captioning, against the same 20-image set: 672.7ms median latency, 2.05GB
peak VRAM, 231M params. Not scored against the Track B rubric — it's a
capability-mismatch case, not a quality one — but the result shows it could
still serve as a cheap scene-description signal for a downstream system even
without direct emotion reasoning. Full output:
`results/eval/track_b_comparison.md` §"Native-capability results".

| Model | Emotion correctness (avg /5) | Contextual grounding (avg /5) | Hallucinations (total, 20 imgs) | Usefulness (avg /5) | Median latency (ms) | Peak VRAM (MB) |
|---|---|---|---|---|---|---|
| Moondream2 (fp16) | 4.20 | **4.45** | **4** | 4.20 | **5,178** | 4,487 |
| Qwen2.5-VL-3B-Instruct (4-bit nf4) | 4.20 | 4.35 | 6 | 4.10 | 11,428 | **2,678** |

Full per-image rubric: `results/eval/track_b_rubric_scores.csv`. Raw model
outputs: `results/logs/moondream2_*.jsonl`, `results/logs/qwen25vl3b_4bit_*.jsonl`.

**Reading the results:** the two models are close on quality — both handled
the "exertion-face could be misread as anger" and "sun-squint could be misread
as displeasure" test cases correctly, showing genuine scene-context reasoning,
not just face reading. Where they consistently *failed together* is the most
useful finding: both missed the Santa hat and reindeer-antlered dog in image
`020`, defaulting to a generic "playful kitchen scene" read and inventing
objects that aren't there (a "screwdriver", a "step stool") — a real
context-grounding gap on a highly specific visual cue, not a case that favors
one model over the other.

Moondream2 is smaller, faster (2.2x), and had fewer hallucinations, but uses
~1.8GB more VRAM at fp16 than the 4-bit Qwen. **Under this plan's
selective-trigger usage pattern** (Track B invoked only when the user speaks,
not continuously), Qwen's slower latency matters less than it would for
Track A's continuous use case — VRAM headroom for concurrent processes is
likely the more decisive factor, which favors Qwen2.5-VL-3B (4-bit) at
2.68GB vs. Moondream2's 4.49GB.

## 3. Decision matrix (per plan §5, in priority order)

1. **VRAM fit (hard filter):** all 5 Track A models and both Track B models
   fit the 6GB budget with room for concurrent processes. Track A models are
   trivially cheap (<1GB); Track B models cost 2.7–4.5GB, a real
   consideration if anything else needs the GPU concurrently.
2. **Accuracy/rubric score:** Track A — DeepFace leads (56.3%/0.547
   macro-F1). Track B — Moondream2 and Qwen2.5-VL-3B are statistically close;
   Moondream2 edges ahead on grounding and hallucination rate.
3. **Latency:** Track A — `fer` is fastest by a wide margin (0.68ms median)
   but least accurate; DeepFace's 8.1ms is still far inside a real-time
   budget. Track B — Moondream2 is 2.2x faster than 4-bit Qwen, consistent
   with the plan's point that Track B latency matters less given
   selective-trigger usage.
4. **License:** now covered separately in
   [`license_comparison.md`](license_comparison.md), not scored here (matches
   plan §6: no model was pre-selected).

**No final pick is being made here** — this is real measured data to weigh
against license terms and the concurrent-load budget, per the plan's own
framing.

## 4. What's still missing (follow-ups, not done this pass)

- Track B staged photos + human-authored ground truth (needs a human
  tester, not an AI standing in for one — see `data/track_b/README.md`).
- PaliGemma-mix, MiniCPM-V 2.6, LLaVA-1.5-7B (not attempted).
- OpenFace-as-classifier (no built-in emotion label, would need a manual
  FACS→emotion mapping — not attempted).

Since this section was last written, the Track A occlusion/lighting stress
set, the license comparison, Py-Feat, and Florence-2/MediaPipe's
native-capability runs have all since been completed — see
[`track_a_stress_test_results.md`](track_a_stress_test_results.md),
[`license_comparison.md`](license_comparison.md), and §1/§2 above.
