"""Builds the unified test sample used to compare FER and VLM models on the
SAME images with the SAME accuracy metric, instead of Track A's full FER2013
split and Track B's separate 20-image scene set.

The FER2013 test manifest is already shuffled, so the first N rows are
already a random sample - no re-shuffling needed here.

Run: .venv/Scripts/python src/eval/build_unified_sample.py [n]
"""
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = REPO_ROOT / "data" / "fer2013" / "test_manifest_shuffled.csv"
OUT_DIR = REPO_ROOT / "data" / "unified_eval"
OUT_MANIFEST = OUT_DIR / "manifest.csv"

DEFAULT_N = 150


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    rows = list(csv.DictReader(open(SOURCE_MANIFEST, encoding="utf-8")))[:n]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
