#!/usr/bin/env python3
"""latency_jitter_test.py - Latency and jitter measurement for HPC networks.

Runs repeated latency probes between nodes using ib_send_lat or ping,
then computes statistics including mean, p99, and jitter.
"""

import subprocess
import sys
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional

NUM_SAMPLES = 100
MESSAGE_SIZE = 64  # bytes


@dataclass
class LatencyResult:
    """Holds latency measurement statistics."""
    samples: List[float]
    mean_us: float
    median_us: float
    p99_us: float
    stddev_us: float
    jitter_us: float  # max - min


def run_ib_latency(target: str, count: int = NUM_SAMPLES, size: int = MESSAGE_SIZE) -> Optional[List[float]]:
    """Run ib_send_lat and extract per-iteration latencies."""
    try:
        result = subprocess.run(
            ["ib_send_lat", target, "-n", str(count), "-s", str(size), "--report_per_iteration"],
            capture_output=True, text=True, timeout=60
        )
        latencies = []
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    latencies.append(float(parts[-1]))
                except ValueError:
                    continue
        return latencies if latencies else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def run_ping_latency(target: str, count: int = NUM_SAMPLES) -> Optional[List[float]]:
    """Fallback: use ICMP ping to measure latency."""
    try:
        result = subprocess.run(
            ["ping", "-c", str(count), "-i", "0.1", target],
            capture_output=True, text=True, timeout=60
        )
        latencies = []
        for line in result.stdout.splitlines():
            if "time=" in line:
                val = line.split("time=")[1].split()[0]
                latencies.append(float(val) * 1000)  # ms to us
        return latencies if latencies else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def compute_stats(latencies: List[float]) -> LatencyResult:
    """Compute latency statistics from raw samples."""
    sorted_lat = sorted(latencies)
    p99_idx = int(len(sorted_lat) * 0.99)
    return LatencyResult(
        samples=latencies,
        mean_us=statistics.mean(latencies),
        median_us=statistics.median(latencies),
        p99_us=sorted_lat[min(p99_idx, len(sorted_lat) - 1)],
        stddev_us=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        jitter_us=max(latencies) - min(latencies),
    )


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <target_host> [num_samples]")
        sys.exit(1)

    target = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_SAMPLES

    print(f"Testing latency to {target} ({count} samples, {MESSAGE_SIZE}B messages)")
    print("Attempting ib_send_lat...")
    latencies = run_ib_latency(target, count)
    if not latencies:
        print("ib_send_lat unavailable, falling back to ping...")
        latencies = run_ping_latency(target, count)
    if not latencies:
        print("ERROR: Could not measure latency.", file=sys.stderr)
        sys.exit(1)

    result = compute_stats(latencies)
    print(f"\n{'='*40}")
    print(f"  Samples:  {len(result.samples)}")
    print(f"  Mean:     {result.mean_us:.2f} us")
    print(f"  Median:   {result.median_us:.2f} us")
    print(f"  P99:      {result.p99_us:.2f} us")
    print(f"  Stddev:   {result.stddev_us:.2f} us")
    print(f"  Jitter:   {result.jitter_us:.2f} us")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
