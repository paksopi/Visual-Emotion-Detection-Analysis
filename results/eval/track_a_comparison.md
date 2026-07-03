# Track A comparison (FER2013 test split)

| Model | N | Accuracy | Macro-F1 | Median latency (ms) | p95 latency (ms) | Peak VRAM/RAM (MB) |
|---|---|---|---|---|---|---|
| DeepFace | 7178 | 0.563 | 0.547 | 8.12 | 10.76 | 368 |
| HSEmotion / EmotiEffLib | 7178 | 0.527 | 0.499 | 8.54 | 10.85 | 368 |
| EfficientFace (RAF-DB checkpoint) | 6178 | 0.525 | 0.450 | 11.07 | 15.51 | 17 |
| fer (justinshenk/fer, mini-xception weights) | 7178 | 0.490 | 0.428 | 0.68 | 0.97 | 342 |
| Py-Feat (Detectorv1) | 7178 | 0.489 | 0.427 | 101.70 | 109.98 | 932 |

## Confusion matrices (rows=true, cols=predicted)

### DeepFace

| true\pred | angry | disgust | fear | happy | sad | surprise | neutral |
|---|---|---|---|---|---|---|---|
| angry | 406 | 10 | 138 | 77 | 166 | 22 | 139 |
| disgust | 31 | 48 | 13 | 3 | 10 | 0 | 6 |
| fear | 93 | 3 | 423 | 61 | 208 | 83 | 153 |
| happy | 50 | 3 | 86 | 1354 | 85 | 32 | 164 |
| sad | 117 | 5 | 188 | 112 | 522 | 17 | 286 |
| surprise | 22 | 1 | 91 | 49 | 26 | 591 | 51 |
| neutral | 76 | 3 | 138 | 117 | 187 | 18 | 694 |

### EfficientFace (RAF-DB checkpoint)

| true\pred | angry | disgust | fear | happy | sad | surprise | neutral |
|---|---|---|---|---|---|---|---|
| angry | 271 | 67 | 38 | 52 | 119 | 81 | 194 |
| disgust | 21 | 35 | 3 | 10 | 19 | 2 | 5 |
| fear | 87 | 28 | 91 | 71 | 220 | 150 | 223 |
| happy | 43 | 4 | 31 | 1156 | 85 | 94 | 134 |
| sad | 63 | 25 | 16 | 57 | 464 | 51 | 383 |
| surprise | 15 | 1 | 41 | 51 | 21 | 542 | 49 |
| neutral | 24 | 15 | 8 | 122 | 158 | 55 | 683 |

> Note: RAF-DB label-index order assumed, not confirmed against upstream training code

### fer (justinshenk/fer, mini-xception weights)

| true\pred | angry | disgust | fear | happy | sad | surprise | neutral |
|---|---|---|---|---|---|---|---|
| angry | 460 | 1 | 38 | 58 | 199 | 38 | 164 |
| disgust | 47 | 12 | 2 | 10 | 26 | 4 | 10 |
| fear | 210 | 4 | 131 | 73 | 302 | 122 | 182 |
| happy | 138 | 1 | 17 | 1164 | 132 | 80 | 242 |
| sad | 156 | 2 | 50 | 85 | 540 | 35 | 379 |
| surprise | 48 | 1 | 83 | 34 | 60 | 538 | 67 |
| neutral | 122 | 1 | 33 | 91 | 296 | 21 | 669 |

### HSEmotion / EmotiEffLib

| true\pred | angry | disgust | fear | happy | sad | surprise | neutral |
|---|---|---|---|---|---|---|---|
| angry | 440 | 68 | 194 | 12 | 78 | 45 | 117 |
| disgust | 20 | 64 | 10 | 2 | 11 | 1 | 1 |
| fear | 129 | 17 | 390 | 18 | 216 | 66 | 181 |
| happy | 13 | 19 | 110 | 1312 | 33 | 124 | 105 |
| sad | 156 | 21 | 144 | 24 | 540 | 24 | 324 |
| surprise | 14 | 2 | 421 | 18 | 9 | 362 | 5 |
| neutral | 103 | 8 | 86 | 58 | 169 | 68 | 676 |

> Note: model has an 8th class (contempt) not present in FER2013 ground truth

### Py-Feat (Detectorv1)

| true\pred | angry | disgust | fear | happy | sad | surprise | neutral |
|---|---|---|---|---|---|---|---|
| angry | 556 | 49 | 39 | 40 | 105 | 78 | 80 |
| disgust | 47 | 36 | 1 | 5 | 15 | 3 | 4 |
| fear | 263 | 12 | 133 | 67 | 241 | 157 | 123 |
| happy | 101 | 24 | 26 | 1272 | 67 | 171 | 72 |
| sad | 296 | 20 | 36 | 58 | 461 | 91 | 212 |
| surprise | 99 | 4 | 166 | 45 | 22 | 441 | 37 |
| neutral | 234 | 12 | 21 | 94 | 217 | 116 | 512 |


## Native-capability results (no emotion output)

Models with no built-in emotion label, run on their actual native task instead (see each runner's docstring).

### emotiefflib_engagement

- Capability: predict_engagement() - sliding-window (default 128 frames) attention classifier over per-frame EfficientNet features, binary engaged/disengaged
- N images: 200
- Median latency: 9.76ms, p95: 11.68ms
- Note: Not an emotion/engagement accuracy benchmark - FER2013 has no video sequences, so there is no meaningful ground truth (or meaningful input) for a sliding-window engagement model here. 'latency' above is genuine per-frame feature-extraction cost (real-time-relevant); the engagement classification call itself was only measured for its own compute cost on a synthetic buffer, see engagement_classification_note. A real deployment needs a live per-subject frame buffer (e.g. the last few seconds of webcam frames), not single unrelated images.

### mediapipe

- Capability: face landmarks + 52 ARKit-style blendshape coefficients (no built-in emotion label)
- N images: 7178
- Face detection rate: 0.850
- Median latency: 8.83ms, p95: 10.94ms
- Note: This is not an emotion-accuracy benchmark -- MediaPipe has no emotion output. The per-label blendshape means show which facial-movement coefficients are most active on average for images from each FER2013 folder, as a proxy for whether its landmark geometry tracks emotion-adjacent expression at all.