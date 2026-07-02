"""Per-call latency tracking: median + p95 over accumulated samples."""
import statistics
import time
from contextlib import contextmanager


class LatencyTimer:
    def __init__(self):
        self.samples_ms: list[float] = []

    @contextmanager
    def measure(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.samples_ms.append((time.perf_counter() - start) * 1000)

    def summary(self) -> dict:
        if not self.samples_ms:
            return {"n": 0, "median_ms": None, "p95_ms": None, "mean_ms": None}
        sorted_samples = sorted(self.samples_ms)
        n = len(sorted_samples)
        p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
        return {
            "n": n,
            "median_ms": statistics.median(sorted_samples),
            "p95_ms": sorted_samples[p95_idx],
            "mean_ms": statistics.mean(sorted_samples),
        }
