"""Sample GPU utilization / memory via nvidia-smi."""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass
class GpuSample:
    t: float
    util_pct: float
    mem_used_mib: float
    mem_total_mib: float


@dataclass
class GpuSampler:
    """Background nvidia-smi poller. No-op if nvidia-smi is missing."""

    interval_s: float = 0.5
    samples: list[GpuSample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    available: bool = True
    error: str | None = None

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        if not self._probe():
            self.available = False
            return
        self.available = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        return self.summary()

    def _probe(self) -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            self.error = str(exc)
            return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._query()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def _query(self) -> GpuSample | None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            self.error = str(exc)
            return None
        line = out.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None
        return GpuSample(
            t=time.perf_counter(),
            util_pct=float(parts[0]),
            mem_used_mib=float(parts[1]),
            mem_total_mib=float(parts[2]),
        )

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {
                "available": self.available,
                "error": self.error,
                "n_samples": 0,
                "gpu_util_mean_pct": None,
                "gpu_util_max_pct": None,
                "mem_used_mean_mib": None,
                "mem_total_mib": None,
            }
        utils = [s.util_pct for s in self.samples]
        mems = [s.mem_used_mib for s in self.samples]
        return {
            "available": True,
            "error": None,
            "n_samples": len(self.samples),
            "gpu_util_mean_pct": mean(utils),
            "gpu_util_max_pct": max(utils),
            "mem_used_mean_mib": mean(mems),
            "mem_total_mib": self.samples[-1].mem_total_mib,
        }
