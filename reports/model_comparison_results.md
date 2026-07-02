# Visual Emotion Detection — Evaluation Results

Status: **first real run complete**, following the protocol in
[`evaluation_plan.md`](evaluation_plan.md). All numbers below come from actual
runs on this machine (RTX 3050 6GB Laptop GPU, Python 3.12 venv) — see
`results/eval/*_summary.json` and `results/logs/*.jsonl` for raw data, and
`src/cv/`, `src/vlm/`, `src/eval/` for the runner/harness code.

## 0. What's real vs. substituted here

- **Track A** ran on the actual FER2013 test split (7,178 images, pulled from
  the `clip-benchmark/wds_fer2013` HF mirror) — this is the real benchmark the
  plan called for.
- Track A's **self-collected stress set** (webcam frames under occlusion/side
  lighting) was **not** produced — no camera available in this environment.
  Face-detection overhead is likewise not measured, since FER2013 images are
  pre-cropped 48×48 faces with no scene for a detector to search (see
  `src/cv/run_fer.py` docstring).
- **Track B**'s image set is a substitute: 20 real photos sourced from the
  EMOTIC/MS-COCO image corpus (not staged), with ground truth **authored by
  the AI assistant that ran this evaluation**, not a human tester. See
  `data/track_b/README.md`. Treat Track B's numbers as illustrative of
  methodology, not a validated benchmark.
- Py-Feat and OpenFace-as-classifier were not run (see §1).

## 1. Track A — CV/FER models (quantitative)

4 of the 6 in-scope candidates ran (MediaPipe/OpenFace excluded per plan
§4 — no built-in emotion label). Py-Feat and a from-scratch EfficientFace
attempt were both "best-effort" per the plan; EfficientFace succeeded,
Py-Feat did not.

**Py-Feat: blocked.** Fails to import — its video-decoding dependency
(`torchcodec`) can't find shared FFmpeg libraries on this machine (the
installed FFmpeg build is statically linked, no `avutil-*.dll` etc. to find).
Not pursued further given the plan's "best-effort" scoping.

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
- All four models struggle most on **fear**, frequently confusing it with
  sad or neutral (see confusion matrices) — consistent with fear being
  FER2013's well-known weak class.
- **VRAM is a non-issue for this whole track**: all four sit under 400MB,
  comfortably inside even a fraction of the 6GB budget. Latency (sub-15ms
  even at p95) is what actually differentiates them for a real-time
  perception layer.

## 2. Track B — VLMs (qualitative + resource)

Both in-budget, non-quantization-required-at-fp16 candidates ran: Moondream2
(fp16) and Qwen2.5-VL-3B-Instruct (4-bit nf4, per the plan's quantization
note). Florence-2, PaliGemma, MiniCPM-V, and LLaVA were not attempted this
pass (time/scope).

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

1. **VRAM fit (hard filter):** all 4 Track A models and both Track B models
   fit the 6GB budget with room for concurrent processes. Track A models are
   trivially cheap (<400MB); Track B models cost 2.7–4.5GB, a real
   consideration if anything else needs the GPU concurrently.
2. **Accuracy/rubric score:** Track A — DeepFace leads (56.3%/0.547
   macro-F1). Track B — Moondream2 and Qwen2.5-VL-3B are statistically close;
   Moondream2 edges ahead on grounding and hallucination rate.
3. **Latency:** Track A — `fer` is fastest by a wide margin (0.68ms median)
   but least accurate; DeepFace's 8.1ms is still far inside a real-time
   budget. Track B — Moondream2 is 2.2x faster than 4-bit Qwen, consistent
   with the plan's point that Track B latency matters less given
   selective-trigger usage.
4. **License:** not evaluated this pass — flagged as follow-up, not scored
   here (matches plan §6: no model was pre-selected).

**No final pick is being made here** — this is real measured data to weigh
against license terms and the concurrent-load budget, per the plan's own
framing.

## 4. What's still missing (follow-ups, not done this pass)

- Track A self-collected occlusion/lighting stress set (needs a camera).
- Track B staged photos + human-authored ground truth (needs a human
  tester, not an AI standing in for one).
- Py-Feat (blocked on FFmpeg/torchcodec on this machine — would need a
  full shared-library FFmpeg build, not the static one currently installed).
- Florence-2, PaliGemma-mix, MiniCPM-V 2.6, LLaVA-1.5-7B (not attempted).
- License comparison across the shortlisted models.
