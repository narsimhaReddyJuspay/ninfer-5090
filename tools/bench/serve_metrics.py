#!/usr/bin/env python3
"""GPU/CPU utilization sampling for serving runs (stdlib only).

Used on the GPU host by ``tools/bench/run_serve_warm_probe.py`` and inside the
RunPod worker image (``deploy/runpod/``) for periodic utilization logging. GPU
samples come from ``nvidia-smi``; CPU utilization from ``/proc/stat`` deltas.
Missing sources degrade to ``None`` instead of failing, so the same code runs
on Linux GPU hosts, containers, and macOS dev machines.
"""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, TextIO, Tuple

GPU_FIELDS: Tuple[str, ...] = ("gpu_util_pct", "gpu_mem_mib", "gpu_power_w", "gpu_temp_c")


def read_gpu(gpu_index: int = 0) -> Optional[Dict[str, Optional[float]]]:
    """One nvidia-smi sample, or None when no GPU/CLI is reachable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    if not lines:
        return None
    parts = lines[0].split(",")
    if len(parts) != len(GPU_FIELDS):
        return None

    def _num(raw: str) -> Optional[float]:
        try:
            return float(raw.strip())
        except ValueError:
            return None  # e.g. power "[N/A]" on idle GPUs

    values = {name: _num(raw) for name, raw in zip(GPU_FIELDS, parts)}
    return values if any(v is not None for v in values.values()) else None


def read_driver(gpu_index: int = 0) -> Optional[Dict[str, str]]:
    """GPU name and driver version, for boot-time logging on GPU hosts."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    if not lines:
        return None
    parts = [part.strip() for part in lines[0].split(",", 1)]
    if len(parts) != 2:
        return None
    return {"name": parts[0], "driver_version": parts[1]}


class CpuTracker:
    """CPU utilization between consecutive /proc/stat reads (Linux only)."""

    def __init__(self) -> None:
        self._prev = self._jiffies()

    @staticmethod
    def _jiffies() -> Optional[Tuple[int, int]]:
        try:
            with open("/proc/stat", encoding="ascii") as handle:
                fields = handle.readline().split()[1:8]
            values = [int(field) for field in fields]
        except (OSError, ValueError):
            return None
        return values[3] + values[4], sum(values)  # (idle+wait, total)

    def utilization(self) -> Optional[float]:
        current = self._jiffies()
        previous, self._prev = self._prev, current
        if previous is None or current is None or current[1] <= previous[1]:
            return None
        idle = current[0] - previous[0]
        total = current[1] - previous[1]
        return max(0.0, min(100.0, 100.0 * (1.0 - idle / total)))


@dataclass
class Sample:
    t: float  # seconds since sampler start
    cpu_util_pct: Optional[float]
    gpu: Optional[Dict[str, Optional[float]]]

    def flat(self) -> Dict[str, Optional[float]]:
        row: Dict[str, Optional[float]] = {"t": self.t, "cpu_util_pct": self.cpu_util_pct}
        gpu = self.gpu or {}
        for name in GPU_FIELDS:
            row[name] = gpu.get(name)
        return row


@dataclass
class Summary:
    duration_s: float
    count: int
    # metric -> {mean, p50, p95, max} over non-None samples, or None when absent
    stats: Dict[str, Optional[Dict[str, float]]] = field(default_factory=dict)

    def format(self) -> str:
        lines = [f"metrics over {self.duration_s:.1f}s ({self.count} samples)"]
        for name, values in self.stats.items():
            if values is None:
                lines.append(f"  {name:14s} unavailable")
            else:
                lines.append(
                    f"  {name:14s} mean {values['mean']:7.1f}  p95 {values['p95']:7.1f}"
                    f"  max {values['max']:7.1f}"
                )
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, object]:
        return {"duration_s": self.duration_s, "count": self.count, "stats": self.stats}


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Percentile of an already-sorted sequence; shared by the bench harnesses."""
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]


class MetricsSampler:
    """Background thread that samples GPU and CPU utilization."""

    COLUMNS = ("t", "cpu_util_pct") + GPU_FIELDS

    def __init__(self, interval_s: float = 0.5, gpu_index: int = 0) -> None:
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self._samples: List[Sample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tracker = CpuTracker()
        self._start_monotonic = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._tracker = CpuTracker()
        self._start_monotonic = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._samples.append(
                Sample(
                    t=time.monotonic() - self._start_monotonic,
                    cpu_util_pct=self._tracker.utilization(),
                    gpu=read_gpu(self.gpu_index),
                )
            )

    def stop(self) -> Summary:
        self._stop.set()
        if self._thread is not None:
            # One in-flight nvidia-smi call can take up to ~10s; never abandon it
            # mid-sample or the summary and CSV would disagree by one row.
            self._thread.join(timeout=max(self.interval_s * 4 + 5, 15))
        duration = time.monotonic() - self._start_monotonic if self._samples else 0.0
        stats: Dict[str, Optional[Dict[str, float]]] = {}
        for name in ("cpu_util_pct", *GPU_FIELDS):
            values = sorted(
                sample.flat()[name] for sample in self._samples
                if sample.flat()[name] is not None
            )
            if not values:
                stats[name] = None
                continue
            stats[name] = {
                "mean": sum(values) / len(values),
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
                "max": values[-1],
            }
        return Summary(duration_s=duration, count=len(self._samples), stats=stats)

    def write_csv(self, path: str) -> int:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.COLUMNS))
            writer.writeheader()
            for sample in self._samples:
                writer.writerow(sample.flat())
        return len(self._samples)


def format_sample(sample: Sample) -> str:
    """One-line rendering for periodic worker logging."""
    gpu = sample.gpu or {}
    parts = [f"cpu={sample.cpu_util_pct:.0f}%" if sample.cpu_util_pct is not None else "cpu=n/a"]
    util = gpu.get("gpu_util_pct")
    mem = gpu.get("gpu_mem_mib")
    power = gpu.get("gpu_power_w")
    temp = gpu.get("gpu_temp_c")

    def _fmt(value: Optional[float], suffix: str) -> str:
        return f"{value:.0f}{suffix}" if value is not None else "n/a"

    parts.append(f"gpu_util={_fmt(util, '%')}")
    parts.append(f"gpu_mem={_fmt(mem, 'MiB')}")
    parts.append(f"gpu_power={_fmt(power, 'W')}")
    parts.append(f"gpu_temp={_fmt(temp, 'C')}")
    return " ".join(parts)


def read_once(gpu_index: int = 0, tracker: Optional[CpuTracker] = None) -> Sample:
    """Single sample for ad-hoc logging; CPU delta is since the previous call."""
    tracker = tracker or CpuTracker()
    return Sample(t=0.0, cpu_util_pct=tracker.utilization(), gpu=read_gpu(gpu_index))
