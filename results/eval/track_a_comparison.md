# Track A comparison (FER2013 test split)

| Model | N | Accuracy | Macro-F1 | Median latency (ms) | p95 latency (ms) | Peak VRAM/RAM (MB) |
|---|---|---|---|---|---|---|
| DeepFace | 7178 | 0.563 | 0.547 | 8.12 | 10.76 | 368 |
| HSEmotion / EmotiEffLib | 7178 | 0.527 | 0.499 | 8.54 | 10.85 | 368 |
| EfficientFace (RAF-DB checkpoint) | 6178 | 0.525 | 0.450 | 11.07 | 15.51 | 17 |
| fer (justinshenk/fer, mini-xception weights) | 7178 | 0.490 | 0.428 | 0.68 | 0.97 | 342 |

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
