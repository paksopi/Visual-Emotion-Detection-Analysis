"""Shared per-model run logging: raw per-call records to results/logs/,
aggregated summary to results/eval/, per reports/evaluation_plan.md section 4.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "results" / "logs"
EVAL_DIR = REPO_ROOT / "results" / "eval"


class RunLogger:
    def __init__(self, model_name: str, track: str):
        self.model_name = model_name
        self.track = track
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOGS_DIR / f"{model_name}_{self.run_id}.jsonl"
        self._fh = open(self.log_path, "a", encoding="utf-8")

    def record(self, **fields):
        row = {"ts": datetime.now(timezone.utc).isoformat(), **fields}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()

    def write_summary(self, summary: dict):
        summary = {
            "model": self.model_name,
            "track": self.track,
            "run_id": self.run_id,
            **summary,
        }
        out_path = EVAL_DIR / f"{self.model_name}_{self.run_id}_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return out_path
