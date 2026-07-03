# Unified model comparison

Every eligible model (FER and VLM together) run on the SAME sample of unified FER2013 sample (n=150), scored by the same accuracy metric - see `src/eval/build_unified_sample.py` and `src/eval/methods.py::UnifiedAccuracyMethod`. VLM answers are short/one-word (capped token count) and mapped to the closest FER2013 label via `src/eval/label_mapping.py` before scoring.

| Model | Type | Test | Speed (median ms) | Accuracy |
|---|---|---|---|---|
| DeepFace | FER | unified FER2013 sample (n=150) | 7.78 | 0.587 |
| HSEmotion / EmotiEffLib | FER | unified FER2013 sample (n=150) | 8.63 | 0.533 |
| fer (mini-xception) | FER | unified FER2013 sample (n=150) | 0.56 | 0.507 |
| PaliGemma-mix-224 (fast, one-word emotion) | VLM | unified FER2013 sample (n=150) | 225.06 | 0.500 |
| Moondream2 (fast, one-word emotion) | VLM | unified FER2013 sample (n=150) | 439.34 | 0.440 |

## Unlisted models

Models that fail capability, license, or VRAM-budget criteria (see `ref/visual_emotion_detection_models.md` §Selection criteria) never get a benchmark run - no accuracy/latency numbers exist for them anywhere in this repo.

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