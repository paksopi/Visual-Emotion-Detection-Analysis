"""Track C: data-driven unified comparison across every registered model
(src/eval/model_registry.py), pulling latency/VRAM, accuracy or rubric score,
real-time session-fitness verdict, and license/production-eligibility into
one table - reports/production_candidate_comparison.md - to pick ONE model
for the live-tutoring perception layer.

Existing aggregate_track_a.py/aggregate_track_b.py and their reports are
untouched by this script (additive layer, not a replacement).
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.aggregate_common import EVAL_DIR, pick_best_summary, rubric_summary_for_model, session_fitness_for_model  # noqa: E402
from eval.licenses import LICENSE_REGISTRY  # noqa: E402
from eval.model_registry import MODEL_REGISTRY  # noqa: E402
from eval.session_fitness import classify_latency  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"


def build_row(model_key: str) -> dict:
    adapter = MODEL_REGISTRY.get(model_key)
    display_name = adapter.display_name if adapter else model_key.replace("_", " ").title()
    output_kind = adapter.output_kind if adapter else "unknown"
    license_id = adapter.license_id if adapter else "unknown"
    production_eligible = adapter.production_eligible if adapter else False

    summary = pick_best_summary(model_key)
    row = {
        "model": model_key, "display_name": display_name, "output_kind": output_kind,
        "track": summary.get("track") if summary else None,
        "n": summary.get("n_images", summary.get("n_items")) if summary else None,
        "median_ms": summary.get("latency", {}).get("median_ms") if summary else None,
        "p95_ms": summary.get("latency", {}).get("p95_ms") if summary else None,
        "peak_vram_mb": summary.get("peak_vram_mb") if summary else None,
        "accuracy": summary.get("metrics", {}).get("accuracy") if summary and "metrics" in summary else None,
        "macro_f1": summary.get("metrics", {}).get("macro_f1") if summary and "metrics" in summary else None,
    }

    rubric = rubric_summary_for_model(model_key)
    row["rubric_avg"] = None
    if rubric:
        row["rubric_avg"] = (rubric["emotion_correctness"] + rubric["contextual_grounding"] + rubric["response_usefulness"]) / 3

    fitness = session_fitness_for_model(model_key)
    if fitness is None and row["median_ms"] is not None:
        fitness = classify_latency(row["median_ms"])
    row["fitness_bucket"] = fitness.get("bucket") if fitness else None
    row["fitness_pass"] = fitness.get("verdict_pass") if fitness else None

    lic = LICENSE_REGISTRY.get(license_id, {"name": license_id, "commercial_ok": False})
    row["license_name"] = lic["name"]
    row["commercial_ok"] = lic["commercial_ok"]
    row["production_eligible"] = production_eligible and lic["commercial_ok"]
    return row


def sort_key(row: dict):
    score = row["accuracy"] if row["accuracy"] is not None else (row["rubric_avg"] or 0) / 5
    return (
        -int(bool(row["production_eligible"])),
        -int(bool(row["fitness_pass"])),
        -(score or 0),
    )


def render(rows: list[dict]) -> str:
    eligible = [r for r in rows if r["production_eligible"]]
    unlisted = [r for r in rows if not r["production_eligible"]]

    lines = ["# Production candidate comparison (Track C)\n"]
    lines.append(
        "Production-eligible models registered in `src/eval/model_registry.py`, compared across "
        "latency/VRAM, accuracy or rubric score, and real-time session-fitness (see "
        "`src/eval/session_fitness.py`). License-restricted, incapable, or over-VRAM-budget models "
        "are never given numbers here - see the Unlisted models section below.\n"
    )
    lines.append(
        "| Model | Kind | N | Median (ms) | p95 (ms) | Fitness | Peak VRAM (MB) | Accuracy | "
        "Macro-F1 | Rubric avg (/5) | License |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(eligible, key=sort_key):
        def fmt(v, spec=""):
            return "n/a" if v is None else format(v, spec)
        lines.append(
            f"| {r['display_name']} | {r['output_kind']} | {fmt(r['n'])} | "
            f"{fmt(r['median_ms'], '.2f')} | {fmt(r['p95_ms'], '.2f')} | "
            f"{r['fitness_bucket'] or 'n/a'} | {fmt(r['peak_vram_mb'], '.0f')} | "
            f"{fmt(r['accuracy'], '.3f')} | {fmt(r['macro_f1'], '.3f')} | "
            f"{fmt(r['rubric_avg'], '.2f')} | {r['license_name']} |"
        )

    lines.append("\n## Unlisted models\n")
    lines.append(
        "Fail capability, license, or VRAM-budget criteria - no benchmark numbers exist for these "
        "anywhere in this repo.\n"
    )
    lines.append("| Model | Reason |")
    lines.append("|---|---|")
    for r in sorted(unlisted, key=lambda r: r["display_name"]):
        adapter = MODEL_REGISTRY.get(r["model"])
        reason = adapter.notes if adapter and adapter.notes else (
            "restricted license" if not r["commercial_ok"] else "not production-eligible"
        )
        lines.append(f"| {r['display_name']} | {reason} |")

    shortlist = [r for r in eligible if r["fitness_pass"]]
    lines.append("\n## Shortlist (production-eligible AND real-time-fit)\n")
    if shortlist:
        for r in sorted(shortlist, key=sort_key):
            lines.append(f"- **{r['display_name']}** — {r['fitness_bucket']}, "
                          f"{fmt_val(r['median_ms'])}ms median, license {r['license_name']}")
    else:
        lines.append("(none)")
    lines.append(
        "\n\\* License-eligible reflects license/commercial-use terms ONLY, not real-time fitness - "
        "e.g. Moondream2 is license-eligible but `async_only`, so it does NOT appear in the Shortlist "
        "below (which requires both). The table can't fully resolve accuracy-vs-latency ties on its "
        "own - the final single-model pick is still your call. See "
        "`ref/visual_emotion_detection_models.md` §5 and `reports/license_comparison.md` for the "
        "reasoning behind each exclusion."
    )
    return "\n".join(lines)


def fmt_val(v):
    return "n/a" if v is None else f"{v:.2f}"


def main():
    rows = [build_row(k) for k in sorted(MODEL_REGISTRY)]
    out_path = REPORTS_DIR / "production_candidate_comparison.md"
    out_path.write_text(render(rows), encoding="utf-8")
    print("wrote", out_path)
    for r in sorted(rows, key=sort_key):
        print(f"  {r['display_name']}: fitness={r['fitness_bucket']} prod_eligible={r['production_eligible']}")


if __name__ == "__main__":
    main()
