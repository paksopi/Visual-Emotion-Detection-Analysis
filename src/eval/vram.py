"""VRAM footprint tracking. Uses torch.cuda peak-allocated stats when the model
runs through PyTorch; falls back to polling `nvidia-smi` for non-PyTorch backends
(e.g. DeepFace/TensorFlow) during a sustained run.
"""
import subprocess
import threading
import time


class VRAMTracker:
    def __init__(self, use_torch: bool = True, poll_interval_s: float = 0.2):
        self.use_torch = use_torch
        self.poll_interval_s = poll_interval_s
        self._poll_thread = None
        self._stop = threading.Event()
        self._peak_mb = 0.0

    def __enter__(self):
        if self.use_torch:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                self.use_torch = False
        if not self.use_torch:
            self._stop.clear()
            self._poll_thread = threading.Thread(target=self._poll, daemon=True)
            self._poll_thread.start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            mb = _nvidia_smi_used_mb()
            if mb is not None:
                self._peak_mb = max(self._peak_mb, mb)
            time.sleep(self.poll_interval_s)

    def __exit__(self, *exc):
        if not self.use_torch and self._poll_thread is not None:
            self._stop.set()
            self._poll_thread.join(timeout=2)
        return False

    def peak_mb(self) -> float:
        if self.use_torch:
            try:
                import torch

                if torch.cuda.is_available():
                    return torch.cuda.max_memory_allocated() / (1024**2)
            except ImportError:
                pass
        return self._peak_mb


def _nvidia_smi_used_mb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None
