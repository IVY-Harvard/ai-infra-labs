#!/usr/bin/env python3
"""error_counter_monitor.py - Periodic IB error counter monitoring with alerting.

Uses perfquery to poll InfiniBand port error counters at regular intervals
and triggers alerts when thresholds are exceeded.
"""

import subprocess
import time
import sys
import re
from dataclasses import dataclass, field
from typing import Dict

POLL_INTERVAL = 5  # seconds
ALERT_THRESHOLD = 10  # new errors per interval

COUNTER_NAMES = [
    "SymbolErrorCounter",
    "LinkErrorRecoveryCounter",
    "LinkDownedCounter",
    "PortRcvErrors",
    "PortRcvRemotePhysicalErrors",
    "PortXmitDiscards",
    "PortXmitConstraintErrors",
    "PortRcvConstraintErrors",
    "LocalLinkIntegrityErrors",
    "ExcessiveBufferOverrunErrors",
]


@dataclass
class PortCounters:
    """Stores error counter values for a single IB port."""
    device: str
    port: int
    counters: Dict[str, int] = field(default_factory=dict)


def run_perfquery(port: int = 1) -> Dict[str, int]:
    """Run perfquery and parse error counter values."""
    counters = {}
    try:
        result = subprocess.run(
            ["perfquery", "-x", str(port)],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            for name in COUNTER_NAMES:
                if name in line:
                    match = re.search(r"\.+(0x[0-9a-fA-F]+|\d+)", line)
                    if match:
                        val = match.group(1)
                        counters[name] = int(val, 16) if val.startswith("0x") else int(val)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[ERROR] perfquery failed: {e}", file=sys.stderr)
    return counters


def check_alerts(prev: Dict[str, int], curr: Dict[str, int]) -> list:
    """Compare counter snapshots and return alerts for threshold violations."""
    alerts = []
    for name in COUNTER_NAMES:
        old_val = prev.get(name, 0)
        new_val = curr.get(name, 0)
        delta = new_val - old_val
        if delta >= ALERT_THRESHOLD:
            alerts.append(f"ALERT: {name} increased by {delta} (threshold={ALERT_THRESHOLD})")
    return alerts


def main():
    """Main monitoring loop."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Monitoring IB port {port} every {POLL_INTERVAL}s (threshold={ALERT_THRESHOLD})")
    prev_counters = run_perfquery(port)

    try:
        while True:
            time.sleep(POLL_INTERVAL)
            curr_counters = run_perfquery(port)
            alerts = check_alerts(prev_counters, curr_counters)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            if alerts:
                for alert in alerts:
                    print(f"[{timestamp}] {alert}")
            else:
                print(f"[{timestamp}] All counters nominal.")
            prev_counters = curr_counters
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
