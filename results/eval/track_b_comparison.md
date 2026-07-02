# Track B comparison (20-image scene-context set, see data/track_b/README.md)

| Model | Emotion correctness (avg /5) | Contextual grounding (avg /5) | Hallucinations (total) | Usefulness (avg /5) | Median latency (ms) | Peak VRAM (MB) |
|---|---|---|---|---|---|---|
| Moondream2 (fp16) | 4.20 | 4.45 | 4 | 4.20 | 5178 | 4487 |
| Qwen2.5-VL-3B-Instruct (4-bit nf4) | 4.20 | 4.35 | 6 | 4.10 | 11428 | 2678 |