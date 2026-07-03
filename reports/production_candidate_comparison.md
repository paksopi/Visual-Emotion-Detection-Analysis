# Production candidate comparison (Track C)

Production-eligible models registered in `src/eval/model_registry.py`, compared across latency/VRAM, accuracy or rubric score, and real-time session-fitness (see `src/eval/session_fitness.py`). License-restricted, incapable, or over-VRAM-budget models are never given numbers here - see the Unlisted models section below.

| Model | Kind | N | Median (ms) | p95 (ms) | Fitness | Peak VRAM (MB) | Accuracy | Macro-F1 | Rubric avg (/5) | License |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepFace | closed_set_emotion | 7178 | 6.74 | 8.96 | realtime | 0 | 0.563 | 0.547 | n/a | MIT |
| HSEmotion / EmotiEffLib | closed_set_emotion | 7178 | 8.17 | 10.19 | realtime | 0 | 0.527 | 0.499 | n/a | Apache-2.0 |
| EfficientFace | closed_set_emotion | 6178 | 8.96 | 11.18 | realtime | 17 | 0.525 | 0.450 | n/a | MIT |
| PaliGemma-mix-224 (fast, one-word emotion) | free_text | 150 | 225.06 | 234.69 | realtime | 5636 | 0.500 | 0.369 | n/a | Gemma Terms of Use (Google) |
| fer (mini-xception) | closed_set_emotion | 7178 | 0.46 | 0.58 | realtime | 0 | 0.490 | 0.428 | n/a | MIT |
| Moondream2 (fast, one-word emotion) | free_text | 150 | 439.34 | 548.14 | borderline | 4488 | 0.440 | 0.396 | n/a | Apache-2.0 |
| EmotiEffLib engagement (sliding window) | engagement | 300 | 9.17 | 15.38 | realtime | 0 | n/a | n/a | n/a | Apache-2.0 |
| Moondream2 (fp16) | free_text | 20 | 5139.44 | 6851.10 | async_only | 4487 | n/a | n/a | 4.28 | Apache-2.0 |

## Unlisted models

Fail capability, license, or VRAM-budget criteria - no benchmark numbers exist for these anywhere in this repo.

| Model | Reason |
|---|---|
| Florence-2-base (native captioning only) | UNLISTED - incapable: task-token model, not instruction-tuned - there is no free-text emotion prompt to set for it. Never scored on accuracy; no numbers included in the unified comparison. |
| LLaVA-1.5-7B | UNLISTED - exceeds the 6GB VRAM budget outright. No runner built, no numbers collected. |
| MediaPipe (top blendshape, no emotion label) | UNLISTED - incapable: no built-in emotion label, outputs blendshapes/landmarks only. |
| MiniCPM-V 2.6 (fast, one-word emotion, 4-bit) | UNLISTED - cancelled (2026-07-03, user call, local storage): its remote-code image processor (image_processing_minicpmv.py) raised a shape-mismatch error against this repo's pinned transformers==4.49.0 - a known class of incompatibility (MiniCPM-V-2_6's repo code targets an older transformers release), same pattern as Py-Feat's separate .venv-pyfeat. Would need its own isolated venv with an older transformers pin to run at all; the ~16GB fp16 checkpoint download was deleted to free local disk space and this candidate is not being pursued further. Revisit only if disk space and a dedicated venv are both available. |
| OpenFace 2.x | UNLISTED - incapable AND license-restricted: only outputs Action Units, no built-in emotion label, and CMU's academic/non-commercial license excludes it from shipping regardless. No runner built, no numbers collected. |
| OpenFace 3.0 | UNLISTED - license: adds a real emotion head (fixing 2.x's capability gap) but the CMU-MultiComp-Lab/OpenFace-3.0 LICENSE file is academic/non-profit non-commercial only, same terms as 2.x. Rejected at the license gate before any runner was built. |
| Qwen2.5-VL-3B-Instruct (4-bit nf4) | UNLISTED - license: Qwen Research License, non-commercial. No numbers included in the unified comparison. |
| Qwen2.5-VL-3B-Instruct (fast, one-word emotion) | UNLISTED - license: same model as 'qwen25vl3b_4bit', short one-word-answer prompt - measured ~1.2s/call vs ~10-17s, but still Qwen Research License regardless of speed. |

## Shortlist (production-eligible AND real-time-fit)

- **DeepFace** — realtime, 6.74ms median, license MIT
- **HSEmotion / EmotiEffLib** — realtime, 8.17ms median, license Apache-2.0
- **EfficientFace** — realtime, 8.96ms median, license MIT
- **PaliGemma-mix-224 (fast, one-word emotion)** — realtime, 225.06ms median, license Gemma Terms of Use (Google)
- **fer (mini-xception)** — realtime, 0.46ms median, license MIT
- **Moondream2 (fast, one-word emotion)** — borderline, 439.34ms median, license Apache-2.0
- **EmotiEffLib engagement (sliding window)** — realtime, 9.17ms median, license Apache-2.0

\* License-eligible reflects license/commercial-use terms ONLY, not real-time fitness - e.g. Moondream2 is license-eligible but `async_only`, so it does NOT appear in the Shortlist below (which requires both). The table can't fully resolve accuracy-vs-latency ties on its own - the final single-model pick is still your call. See `ref/visual_emotion_detection_models.md` §5 and `reports/license_comparison.md` for the reasoning behind each exclusion.