# Production candidate comparison (Track C)

Every model registered in `src/eval/model_registry.py`, compared across latency/VRAM, accuracy or rubric score, real-time session-fitness (see `src/eval/session_fitness.py`), and license/production-eligibility (see `src/eval/licenses.py`). Sorted so eligible/fast/accurate candidates surface first - license-excluded or too-slow candidates are still shown below for reference, not omitted.

| Model | Kind | N | Median (ms) | p95 (ms) | Fitness | Peak VRAM (MB) | Accuracy | Macro-F1 | Rubric avg (/5) | License | Commercial OK | License-eligible* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepFace | closed_set_emotion | 7178 | 6.74 | 8.96 | realtime | 0 | 0.563 | 0.547 | n/a | MIT | ✅ | ✅ |
| HSEmotion / EmotiEffLib | closed_set_emotion | 7178 | 8.17 | 10.19 | realtime | 0 | 0.527 | 0.499 | n/a | Apache-2.0 | ✅ | ✅ |
| EfficientFace | closed_set_emotion | 6178 | 8.96 | 11.18 | realtime | 17 | 0.525 | 0.450 | n/a | MIT | ✅ | ✅ |
| fer (mini-xception) | closed_set_emotion | 7178 | 0.46 | 0.58 | realtime | 0 | 0.490 | 0.428 | n/a | MIT | ✅ | ✅ |
| EmotiEffLib engagement (sliding window) | engagement | 300 | 9.17 | 15.38 | realtime | 0 | n/a | n/a | n/a | Apache-2.0 | ✅ | ✅ |
| Moondream2 (fast, one-word emotion) | free_text | 20 | 864.16 | 984.95 | borderline | 4488 | n/a | n/a | n/a | Apache-2.0 | ✅ | ✅ |
| Moondream2 (fp16) | free_text | 20 | 5139.44 | 6851.10 | async_only | 4487 | n/a | n/a | 4.28 | Apache-2.0 | ✅ | ✅ |
| Florence-2-base (native captioning only) | native_other | 20 | 927.50 | 1294.52 | borderline | 2050 | n/a | n/a | n/a | MIT | ✅ | — |
| MediaPipe (top blendshape, no emotion label) | native_other | 7178 | 10.18 | 12.95 | realtime | 0 | n/a | n/a | n/a | Apache-2.0 | ✅ | — |
| Qwen2.5-VL-3B-Instruct (4-bit nf4) | free_text | 20 | 17139.38 | 19098.37 | async_only | 3205 | n/a | n/a | 4.22 | Qwen RESEARCH LICENSE | ⚠️ restricted | — |
| Qwen2.5-VL-3B-Instruct (fast, one-word emotion) | free_text | 20 | 1409.03 | 2078.08 | async_only | 3205 | n/a | n/a | n/a | Qwen RESEARCH LICENSE | ⚠️ restricted | — |

## Shortlist (production-eligible AND real-time-fit)

- **DeepFace** — realtime, 6.74ms median, license MIT
- **HSEmotion / EmotiEffLib** — realtime, 8.17ms median, license Apache-2.0
- **EfficientFace** — realtime, 8.96ms median, license MIT
- **fer (mini-xception)** — realtime, 0.46ms median, license MIT
- **EmotiEffLib engagement (sliding window)** — realtime, 9.17ms median, license Apache-2.0
- **Moondream2 (fast, one-word emotion)** — borderline, 864.16ms median, license Apache-2.0

\* License-eligible reflects license/commercial-use terms ONLY, not real-time fitness - e.g. Moondream2 is license-eligible but `async_only`, so it does NOT appear in the Shortlist below (which requires both). The table can't fully resolve accuracy-vs-latency ties on its own - the final single-model pick is still your call. See `ref/visual_emotion_detection_models.md` §5 and `reports/license_comparison.md` for the reasoning behind each exclusion.