"""Real-time session-fitness method: classifies a model's already-measured
latency against the constraint that actually matters for this repo's target
use case (a per-turn signal inside a live LLM chat session) - not a new
metric, a bucketing pass over numbers LatencyTimer/RunLogger already wrote.

Thresholds are derived from this session's empirical numbers: Track A CV/FER
models measured at 8-12ms land solidly in "realtime"; Track B VLMs measured
at 5.2s-11.4s land solidly in "async_only". "borderline" is a real bucket,
currently expected to stay empty until a 300ms-1s-class model is measured.
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from eval import RunLogger  # noqa: E402

EVAL_DIR = REPO_ROOT / "results" / "eval"

# Ordered (threshold_ms, bucket_name, verdict_pass, description) - first
# matching threshold wins (median_ms <= threshold_ms).
LATENCY_BUCKETS = [
    (300, "realtime", True, "Fully non-blocking per-turn signal - safe to call synchronously in the chat-turn path."),
    (1000, "borderline", True, "Viable async-with-debounce: run in a background thread, update every N turns / few seconds, chat reads the last-known value, never blocks."),
    (float("inf"), "async_only", False, "Too slow to gate a chat turn even with debounce - must run purely out-of-band."),
]


def classify_latency(median_ms: float) -> dict:
    for threshold_ms, bucket, verdict_pass, description in LATENCY_BUCKETS:
        if median_ms <= threshold_ms:
            return {
                "bucket": bucket,
                "verdict_pass": verdict_pass,
                "median_ms": median_ms,
                "threshold_ms": threshold_ms,
                "description": description,
            }
    raise AssertionError("unreachable - last bucket threshold is inf")


def classify_model(model_key: str) -> dict:
    """Find the most recent non-fitness summary for model_key and classify
    it. Used by methods.py's session_fitness Method for single-model runs
    (`run_method.py --model X --method session_fitness`).
    """
    # Match exactly "{model_key}_{run_id}_summary.json" (run_id = UTC
    # %Y%m%dT%H%M%SZ) - a plain glob would also match derived summaries like
    # "{model_key}_live_..." (live-webcam demo) or "{model_key}_fitness_..."
    # (this module's own output), neither of which has a latency field.
    pattern = re.compile(rf"^{re.escape(model_key)}_\d{{8}}T\d{{6}}Z_summary\.json$")
    candidates = sorted(p for p in EVAL_DIR.glob("*_summary.json") if pattern.match(p.name))
    if not candidates:
        raise FileNotFoundError(
            f"No summary found for model '{model_key}' in {EVAL_DIR} - run the "
            f"latency_vram method (or the model's existing runner) first."
        )
    path = candidates[-1]  # most recent run_id, lexicographically sortable (UTC timestamp format)
    data = json.loads(path.read_text(encoding="utf-8"))
    latency = data.get("latency")
    if not latency or latency.get("median_ms") is None:
        raise ValueError(f"{path} has no latency.median_ms to classify")
    return {"source_summary": path.name, "source_track": data.get("track"),
            "latency": latency, **classify_latency(latency["median_ms"])}


def compute_for_all_summaries() -> list[Path]:
    """Scans every results/eval/*_summary.json with a latency.median_ms field
    and writes one companion session-fitness summary per source file. Does
    NOT mutate the original summary files.
    """
    written = []
    for path in sorted(EVAL_DIR.glob("*_summary.json")):
        if "_fitness_" in path.name:
            continue  # don't re-classify our own output
        data = json.loads(path.read_text(encoding="utf-8"))
        latency = data.get("latency")
        if not latency or latency.get("median_ms") is None:
            continue
        classification = classify_latency(latency["median_ms"])
        logger = RunLogger(model_name=f"{data['model']}_fitness", track="session-fitness")
        out = logger.write_summary({
            "source_summary": path.name,
            "source_track": data.get("track"),
            "latency": latency,
            **classification,
        })
        logger.close()
        written.append(out)
        print(f"{data['model']}: {classification['bucket']} (median {latency['median_ms']}ms) -> {out.name}")
    return written


if __name__ == "__main__":
    compute_for_all_summaries()
