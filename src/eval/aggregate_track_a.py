"""Aggregate Track A per-model summary JSONs into one comparison table."""
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "results" / "eval"
REPORTS_DIR = REPO_ROOT / "reports"

MODEL_DISPLAY = {
    "fer_lib": "fer (justinshenk/fer, mini-xception weights)",
    "deepface": "DeepFace",
    "hsemotion": "HSEmotion / EmotiEffLib",
    "efficientface": "EfficientFace (RAF-DB checkpoint)",
    "pyfeat": "Py-Feat (Detectorv1)",
}


def main():
    best_by_model = {}
    native_by_model = {}
    for f in EVAL_DIR.glob("*_summary.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        model = data["model"]
        if data.get("track") == "A-native":
            if model not in native_by_model or data["n_images"] > native_by_model[model]["n_images"]:
                native_by_model[model] = data
            continue
        if "metrics" not in data:
            continue
        if model not in best_by_model or data["n_images"] > best_by_model[model]["n_images"]:
            best_by_model[model] = data

    lines = ["# Track A comparison (FER2013 test split)\n"]
    lines.append(
        "| Model | N | Accuracy | Macro-F1 | Median latency (ms) | p95 latency (ms) | Peak VRAM/RAM (MB) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for model, data in sorted(best_by_model.items(), key=lambda kv: -kv[1]["metrics"]["accuracy"]):
        m = data["metrics"]
        lat = data["latency"]
        lines.append(
            f"| {MODEL_DISPLAY.get(model, model)} | {data['n_images']} | "
            f"{m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
            f"{lat['median_ms']:.2f} | {lat['p95_ms']:.2f} | {data['peak_vram_mb']:.0f} |"
        )

    lines.append("\n## Confusion matrices (rows=true, cols=predicted)\n")
    for model, data in best_by_model.items():
        m = data["metrics"]
        lines.append(f"### {MODEL_DISPLAY.get(model, model)}\n")
        labels = m["labels"]
        header = "| true\\pred | " + " | ".join(labels) + " |"
        sep = "|---" * (len(labels) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for true_label, row in zip(labels, m["confusion_matrix"]):
            lines.append(f"| {true_label} | " + " | ".join(str(v) for v in row) + " |")
        lines.append("")
        if "note" in data:
            lines.append(f"> Note: {data['note']}\n")

    if native_by_model:
        lines.append("\n## Native-capability results (no emotion output)\n")
        lines.append(
            "Models with no built-in emotion label, run on their actual native task instead "
            "(see each runner's docstring)."
        )
        for model, data in native_by_model.items():
            lat = data["latency"]
            lines.append(f"\n### {model}\n")
            lines.append(f"- Capability: {data.get('capability', 'n/a')}")
            lines.append(f"- N images: {data['n_images']}")
            if "face_detection_rate" in data:
                lines.append(f"- Face detection rate: {data['face_detection_rate']:.3f}")
            lines.append(f"- Median latency: {lat['median_ms']:.2f}ms, p95: {lat['p95_ms']:.2f}ms")
            if "note" in data:
                lines.append(f"- Note: {data['note']}")

    out_path = EVAL_DIR / "track_a_comparison.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out_path)
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
