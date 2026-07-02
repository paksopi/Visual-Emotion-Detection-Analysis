# Track B comparison (20-image scene-context set, see data/track_b/README.md)

| Model | Emotion correctness (avg /5) | Contextual grounding (avg /5) | Hallucinations (total) | Usefulness (avg /5) | Median latency (ms) | Peak VRAM (MB) |
|---|---|---|---|---|---|---|
| Moondream2 (fp16) | 4.20 | 4.45 | 4 | 4.20 | 5358 | 4475 |
| Qwen2.5-VL-3B-Instruct (4-bit nf4) | 4.20 | 4.35 | 6 | 4.10 | 10390 | 2678 |

## Native-capability results (no open-ended emotion prompting)


### florence2

- Capability: native task-token dense captioning (no open-ended prompting -- see docstring)
- N images: 20
- Median latency: 672.66ms, p95: 1001.70ms
- Peak VRAM: 2050MB
- Note: Not scored against the Track B emotion rubric -- Florence-2-base cannot take an open-ended emotion prompt at all (confirmed: '<VQA>What emotion...' degenerates to garbled output, not a real answer). This captures what it's actually good at: dense, detailed scene captioning, which a downstream system could still use as a cheap (231M param) scene-description signal even without direct emotion reasoning.