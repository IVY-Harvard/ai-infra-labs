# Lab 10: Bandwidth Optimization

## Overview

This lab explores techniques for maximizing network bandwidth utilization in HPC clusters. You will experiment with adaptive routing, traffic isolation, and QoS configuration to optimize data transfer performance across InfiniBand fabrics.

## Learning Objectives

- Understand adaptive routing strategies in fat-tree topologies
- Configure traffic isolation using partitions and service levels
- Apply QoS policies to prioritize RDMA traffic classes
- Measure the impact of optimization techniques on real workloads

## Lab Structure

| File | Description |
|------|-------------|
| `adaptive_routing.md` | Adaptive routing strategy concepts and configuration |
| `traffic_isolation.md` | Traffic isolation via VLANs, partitions, and SLs |
| `qos_config.sh` | QoS configuration script for IB subnet manager |

## Prerequisites

- Completed Lab 09 (Network Diagnosis)
- Access to a multi-switch IB fabric (or simulator)
- OpenSM or equivalent subnet manager installed
- Root/admin access for QoS policy changes

## Experiments

### Experiment 1: Adaptive Routing
1. Read `adaptive_routing.md` for background
2. Enable adaptive routing on the subnet manager
3. Run `ib_send_bw` with multiple streams and observe path distribution
4. Compare throughput with static vs. adaptive routing

### Experiment 2: Traffic Isolation
1. Read `traffic_isolation.md` for partition key (PKey) concepts
2. Create isolated partitions for compute and storage traffic
3. Verify isolation using `ibdiagnet` partition analysis
4. Measure cross-partition interference

### Experiment 3: QoS Configuration
1. Run `qos_config.sh` to apply service level policies
2. Generate mixed traffic (bulk + latency-sensitive)
3. Verify priority enforcement with `perfquery` SL counters
4. Tune parameters for optimal workload mix

## Expected Results

- Adaptive routing should improve multi-path bandwidth by 20-40%
- Traffic isolation should eliminate interference between traffic classes
- QoS should maintain low latency for priority traffic under congestion
