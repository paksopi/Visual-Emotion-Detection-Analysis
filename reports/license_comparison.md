# License comparison — shortlisted models

Licenses for every model actually shortlisted/benchmarked in this repo (Track
A + Track B), plus the two CV-family models that were surveyed but excluded
for not producing a direct emotion label (MediaPipe, OpenFace) since they
remain relevant to a future perception-layer build.

| Model | License | Commercial use | Source checked |
|---|---|---|---|
| Mini-Xception (`oarriaga/face_classification`) | MIT | ✅ Yes | GitHub repo license |
| `fer` (Python lib, ships Mini-Xception weights) | MIT | ✅ Yes | Installed package metadata (`pip show`) |
| DeepFace | MIT | ✅ Yes | Installed package metadata (`pip show`) |
| HSEmotion / EmotiEffLib (`hsemotion-onnx`) | Apache-2.0 | ✅ Yes | Installed package metadata (`pip show`) |
| EfficientFace | MIT | ✅ Yes | `LICENSE` file in the cloned upstream repo (`data/models/efficientface_repo/LICENSE`) |
| Py-Feat | MIT | ✅ Yes | Installed package metadata (`pip show`) |
| MediaPipe | Apache-2.0 | ✅ Yes (as a tracking framework — not a standalone emotion classifier, see [`ref/visual_emotion_detection_models.md`](../ref/visual_emotion_detection_models.md) §4) | GitHub repo license |
| OpenFace | Custom, non-commercial | ⚠️ **Restricted** — free use is academic/non-commercial only; commercial use requires a separate license via Carnegie Mellon's Flintbox | GitHub repo `OpenFace-license.txt` (also bundles dlib/OpenBLAS/OpenCV, each with their own license) |
| OpenFace 3.0 (`CMU-MultiComp-Lab/OpenFace-3.0`) | Custom, non-commercial (same CMU MultiComp Lab lineage) | ⚠️ **Restricted — identical terms to 2.x.** Checked directly: repo `LICENSE` is CMU's "ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY" agreement. Adds a real direct emotion-recognition head (see `ref/visual_emotion_detection_models.md` §5) but that capability doesn't change the licensing — excluded from this repo's production-candidate harness on license grounds alone. | GitHub repo `LICENSE` file, fetched and read verbatim 2026-07-03 |
| Qwen2.5-VL-3B-Instruct | **"Qwen RESEARCH LICENSE"** (`license_name: qwen-research`) | ⚠️ **Restricted — research use only.** Not Apache-2.0, despite the Qwen2.5-VL *code* repo on GitHub being Apache-2.0 — that only covers the inference code, not these specific model weights. Commercial deployment needs a separate agreement with Alibaba Cloud. | Hugging Face model card YAML frontmatter (`license_name`/`license_link` fields), cross-checked against the actual `LICENSE` file at that link |
| Moondream2 | Apache-2.0 | ✅ Yes | Hugging Face model card |
| EmotiEffLib (`emotiefflib` PyPI package, successor to `hsemotion-onnx`, adds `predict_engagement()`) | Apache-2.0 | ✅ Yes — package README states "no limitation for both academic and commercial usage" | Installed package metadata (`pip show emotiefflib`) + upstream `LICENSE`, checked 2026-07-03 |

## Takeaways for future projects

- **All the CV/FER classifiers actually benchmarked here (Mini-Xception, `fer`,
  DeepFace, HSEmotion, EfficientFace, Py-Feat) are permissively licensed**
  (MIT or Apache-2.0) — safe to ship in a commercial product as-is.
- **OpenFace is not free for commercial use** despite being open-source —
  worth remembering since it's a common recommendation for AU-based work.
  **This carries forward to OpenFace 3.0** (checked 2026-07-03) — a new
  direct emotion head doesn't change CMU's non-commercial license terms.
- **EmotiEffLib's newer `emotiefflib` package (Apache-2.0) adds an
  `predict_engagement()` API** distinct from categorical emotion — worth
  prioritizing for a student-session product over OpenFace 3.0's
  license-restricted emotion head, since it's commercially usable today
  with a library already in this repo's dependency set.
- **Qwen2.5-VL-3B-Instruct is research-only**, which matters a lot given it's
  one of the two Track B models actually benchmarked as a top performer in
  this repo (see `reports/model_comparison_results.md`) — Moondream2
  (Apache-2.0) is the commercially-safe alternative of the two, and won
  Track B's grounding/hallucination metrics anyway.
- Always check the license on the specific checkpoint/weights being used,
  not just the model family's code repository — they can differ, as seen
  with Qwen2.5-VL here.
