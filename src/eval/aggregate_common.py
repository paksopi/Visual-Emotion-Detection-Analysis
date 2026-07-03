"""Shared helpers for aggregate_production_candidates.py only. The existing
aggregate_track_a.py/aggregate_track_b.py duplicate similar logic inline and
are deliberately left unmodified (working scripts, additive-layer rule) -
this is an accepted small duplication, a candidate for a follow-up dedupe
once the new aggregator is proven out.
"""
import csv
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "results" / "eval"
RUBRIC_CSV = EVAL_DIR / "track_b_rubric_scores.csv"


def pick_best_summary(model_key: str, track_filter=None) -> dict | None:
    """Prefer the summary with the most items (n_images/n_items), else the
    most recent run_id, among summaries for model_key. track_filter, if
    given, is a set of acceptable `track` values.
    """
    best = None
    for f in EVAL_DIR.glob(f"{model_key}_*_summary.json"):
        if "_live" in f.name or "_fitness" in f.name:
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("model") != model_key:
            continue
        if track_filter and data.get("track") not in track_filter:
            continue
        n = data.get("n_images", data.get("n_items", 0))
        if best is None or n > best.get("n_images", best.get("n_items", 0)) or (
            n == best.get("n_images", best.get("n_items", 0)) and data["run_id"] > best["run_id"]
        ):
            best = data
    return best


def session_fitness_for_model(model_key: str) -> dict | None:
    candidates = sorted(EVAL_DIR.glob(f"{model_key}_fitness_*_summary.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def rubric_summary_for_model(model_key: str) -> dict | None:
    """Mean of the 4 rubric dimensions across every row for model_key in the
    existing (manually-authored, see data/track_b/README.md) rubric CSV.
    """
    if not RUBRIC_CSV.exists():
        return None
    rows = [r for r in csv.DictReader(open(RUBRIC_CSV, encoding="utf-8")) if r["model"] == model_key]
    if not rows:
        return None
    dims = ["emotion_correctness", "contextual_grounding", "hallucination_count", "response_usefulness"]
    means = {d: statistics.mean(float(r[d]) for r in rows) for d in dims}
    means["n_scored"] = len(rows)
    return means
