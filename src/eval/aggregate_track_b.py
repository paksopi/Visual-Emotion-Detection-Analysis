"""Aggregate Track B rubric scores + resource metrics into one comparison table."""
import csv
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "results" / "eval"
RUBRIC_CSV = EVAL_DIR / "track_b_rubric_scores.csv"

MODEL_DISPLAY = {
    "moondream2": "Moondream2 (fp16)",
    "qwen25vl3b_4bit": "Qwen2.5-VL-3B-Instruct (4-bit nf4)",
}


def main():
    rows = list(csv.DictReader(open(RUBRIC_CSV, encoding="utf-8")))
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    resource = {}
    for f in EVAL_DIR.glob("*_summary.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("track") == "B":
            resource[data["model"]] = data

    lines = ["# Track B comparison (20-image scene-context set, see data/track_b/README.md)\n"]
    lines.append(
        "| Model | Emotion correctness (avg /5) | Contextual grounding (avg /5) | "
        "Hallucinations (total) | Usefulness (avg /5) | Median latency (ms) | Peak VRAM (MB) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for model, scores in by_model.items():
        ec = statistics.mean(int(s["emotion_correctness"]) for s in scores)
        cg = statistics.mean(int(s["contextual_grounding"]) for s in scores)
        hall = sum(int(s["hallucination_count"]) for s in scores)
        use = statistics.mean(int(s["response_usefulness"]) for s in scores)
        res = resource.get(model, {})
        lat = res.get("latency", {})
        lines.append(
            f"| {MODEL_DISPLAY.get(model, model)} | {ec:.2f} | {cg:.2f} | {hall} | {use:.2f} | "
            f"{lat.get('median_ms', float('nan')):.0f} | {res.get('peak_vram_mb', float('nan')):.0f} |"
        )

    out_path = EVAL_DIR / "track_b_comparison.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
