"""Builds reports/unified_comparison.md: one table across every eligible
model (FER and VLM together), all scored on the same sample
(data/unified_eval/manifest.csv, see build_unified_sample.py and
methods.py::UnifiedAccuracyMethod) with the same accuracy metric - answers
"why can't every model take the same test" by making every model take the
same test.

Models that fail the selection criteria (capability / license / VRAM) are
never given numbers here - they're listed once in a separate "Unlisted
models" section with just their exclusion reason. See
ref/visual_emotion_detection_models.md for the full criteria.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.aggregate_common import unified_summary_for_model  # noqa: E402
from eval.model_registry import MODEL_REGISTRY  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"


def build_row(model_key: str) -> dict | None:
    adapter = MODEL_REGISTRY[model_key]
    summary = unified_summary_for_model(model_key)
    if summary is None:
        return None
    return {
        "display_name": adapter.display_name,
        "type": summary.get("model_type", "VLM" if adapter.modality == "scene_rgb" else "FER"),
        "test": summary.get("test", "n/a"),
        "median_ms": summary.get("latency", {}).get("median_ms"),
        "accuracy": summary.get("metrics", {}).get("accuracy"),
    }


def render(rows: list[dict], unlisted: list[tuple[str, str]]) -> str:
    lines = ["# Unified model comparison\n"]
    lines.append(
        "Every eligible model (FER and VLM together) run on the SAME sample of "
        f"{rows[0]['test'] if rows else 'the unified FER2013 sample'}, scored by the same "
        "accuracy metric - see `src/eval/build_unified_sample.py` and "
        "`src/eval/methods.py::UnifiedAccuracyMethod`. VLM answers are short/one-word "
        "(capped token count) and mapped to the closest FER2013 label via "
        "`src/eval/label_mapping.py` before scoring.\n"
    )
    lines.append("| Model | Type | Test | Speed (median ms) | Accuracy |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: -(r["accuracy"] or 0)):
        acc = "n/a" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
        ms = "n/a" if r["median_ms"] is None else f"{r['median_ms']:.2f}"
        lines.append(f"| {r['display_name']} | {r['type']} | {r['test']} | {ms} | {acc} |")

    lines.append("\n## Unlisted models\n")
    lines.append(
        "Models that fail capability, license, or VRAM-budget criteria (see "
        "`ref/visual_emotion_detection_models.md` §Selection criteria) never get a benchmark "
        "run - no accuracy/latency numbers exist for them anywhere in this repo.\n"
    )
    lines.append("| Model | Reason |")
    lines.append("|---|---|")
    for name, reason in unlisted:
        lines.append(f"| {name} | {reason} |")
    return "\n".join(lines)


def main():
    rows, unlisted = [], []
    for key in sorted(MODEL_REGISTRY):
        adapter = MODEL_REGISTRY[key]
        if not adapter.production_eligible:
            unlisted.append((adapter.display_name, adapter.notes or "excluded"))
            continue
        row = build_row(key)
        if row is None:
            print(f"skipping {key}: no unified_accuracy summary yet "
                  f"(run `python src/eval/run_method.py --model {key} --method unified_accuracy`)")
            continue
        rows.append(row)

    out_path = REPORTS_DIR / "unified_comparison.md"
    out_path.write_text(render(rows, unlisted), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
