# Lab 09: Network Diagnosis

## Overview
Systematic approaches to diagnosing high-performance network issues in GPU clusters. Covers link health, error monitoring, and latency analysis.

## Lab Files

| File | Description |
|------|-------------|
| `link_health_check.sh` | Bash script to check link state, speed, FEC errors, and cable health |
| `error_counter_monitor.py` | Continuous monitoring of InfiniBand/RoCE error counters with alerting |
| `latency_jitter_test.py` | Measures P50/P99/P999 latency and jitter across cluster node pairs |

## Common Network Issues in GPU Clusters

### 1. Link Flapping
- **Symptoms**: Intermittent connection drops, training job failures
- **Causes**: Bad cables, loose connectors, FEC threshold exceeded
- **Diagnosis**: `link_health_check.sh` monitors link state transitions

### 2. Silent Data Corruption
- **Symptoms**: NaN loss, incorrect gradients, non-reproducible results
- **Causes**: Bit errors below FEC correction threshold, faulty transceivers
- **Diagnosis**: `error_counter_monitor.py` tracks CRC/symbol error rates

### 3. Latency Spikes
- **Symptoms**: Slow AllReduce, stragglers in synchronous training
- **Causes**: PFC storms, congestion, ECMP imbalance, switch buffer overflow
- **Diagnosis**: `latency_jitter_test.py` identifies problematic node pairs

## Quick Start
```bash
# Check link health on all Mellanox NICs
bash link_health_check.sh

# Start error counter monitoring (runs continuously)
python3 error_counter_monitor.py --interval 10 --threshold 100

# Run latency test between two nodes
python3 latency_jitter_test.py --server 10.0.0.1 --duration 30
```
