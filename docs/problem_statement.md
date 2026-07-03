# Problem statement & production module plan

## Problem

An LLM tutoring product needs a live signal of student emotional/engagement state (frustrated,
confused, disengaged, neutral, positive) from webcam video, to steer the LLM's tone and pacing.
The signal must not block a chat turn, must fit inside a 6GB VRAM budget shared with the LLM
itself, and must be commercially shippable (no research-only licenses).

## Constraints

| Constraint | Value |
|---|---|
| Latency | Must not block a chat turn — realtime (≤300ms) to gate synchronously; anything slower runs out-of-band |
| VRAM | 6GB total, shared with the LLM |
| Accuracy floor | ~0.59 (DeepFace, unified sample) is the current ceiling, not a target to beat |
| License | Commercially shippable — hard-excludes research-only licenses regardless of accuracy |
| Deployment | Local/on-device, single consumer GPU, one session at a time |

See [`README.md`](../README.md) selection criteria and
[`ref/visual_emotion_detection_models.md`](../ref/visual_emotion_detection_models.md) for the
full candidate survey behind these numbers.

## Recommended architecture: hybrid pipeline

- **Primary signal — DeepFace, synchronous, per-frame.** 7.78ms median, MIT, best accuracy
  (0.587) among realtime candidates on the unified sample.
- **Secondary signal — EmotiEffLib engagement (windowed).** Runs alongside DeepFace at near-zero
  extra cost; targets "is this student engaged," arguably more actionable than 7-way emotion.
- **Periodic sanity-check — Moondream2 (fast, one-word), async, every 5-15s.** Never gates a
  turn. Catches cases where DeepFace's face detector is confidently wrong (occlusion, off-angle)
  or cross-validates a persistent state change before the LLM leans on it.
- **Rejected as primary:** any VLM — latency is decode-bound (cost scales with output tokens,
  not image size), architecturally incompatible with per-frame gating regardless of model choice
  or license.

## Not yet built: the production module

Everything above is validated by the survey/benchmark harness (`src/eval/`), but no code wires
the winning models into an actual tutoring-loop signal yet. Planned, not started:

- `src/production/pipeline.py` — runs DeepFace + EmotiEffLib engagement per frame, Moondream2-fast
  on a timer, using the existing predictor factories in `src/cv/live_webcam_demo.py` and
  `src/vlm/live_webcam_vlm_demo.py` (no new model-loading code — reuse what's registered in
  `src/eval/model_registry.py`).
- Debounce/smoothing: majority-vote or EMA over a ~3-5s window (10-20 frames) before a state
  change is exposed to the LLM — raw per-frame CV output is too noisy to pass straight through
  (see the live-eval section in the README, where DeepFace and HSEmotion disagreed on the same
  window).
- Output contract: a compact structured field appended to the LLM's context turn, e.g.
  `{"emotion": "frustrated", "engagement": "low", "confidence": "sustained", "source": "cv"}`,
  with `"source": "vlm_override"` when Moondream2 disagrees with the debounced CV state for two
  consecutive checks (~10-30s) — a single disagreement is treated as noise.
- Fallback state (`"unknown"`) when face detection fails for N consecutive frames — occlusion
  and off-angle are the dominant real-world failure mode (see
  [`reports/track_a_stress_test_results.md`](../reports/track_a_stress_test_results.md): 0/3
  face-detect under occlusion, 1/3 under off-angle), not misclassification.

## Open risks

- No end-to-end live test of the full hybrid loop yet — only individual pieces are validated.
- FER2013-derived accuracy is a lab-photo ceiling; real webcam sessions (poor lighting, side
  angles) will likely score lower. Don't present the signal as ground truth to the student.
- VRAM headroom under real load (hybrid pipeline + LLM together) hasn't been measured on the
  actual 6GB target GPU.
