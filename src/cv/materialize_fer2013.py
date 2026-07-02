"""One-off: materialize FER2013 test images from the HF mirror
(clip-benchmark/wds_fer2013) to disk, matching data/fer2013/test_manifest.csv's
paths (data/fer2013/test/<label>/<global_index:05d>.jpg). The dataset's row
order already matches the manifest's global index exactly (grouped by class,
verified against the manifest's per-label row-count breakpoints).
"""

from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "fer2013" / "test"
CLASSNAMES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


def main():
    ds = load_dataset("clip-benchmark/wds_fer2013", split="test")
    for label in CLASSNAMES:
        (OUT_DIR / label).mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(ds):
        label = CLASSNAMES[row["cls"]]
        out_path = OUT_DIR / label / f"{i:05d}.jpg"
        row["jpg"].save(out_path)
        if i % 1000 == 0:
            print(f"{i}/{len(ds)}")

    print("done")


if __name__ == "__main__":
    main()
