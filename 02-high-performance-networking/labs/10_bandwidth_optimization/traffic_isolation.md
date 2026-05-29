# Traffic Isolation in HPC Networks

## Overview

Traffic isolation prevents different workload classes from interfering with each other on shared network fabric. In InfiniBand, isolation is achieved through Partition Keys (PKeys), Service Levels (SLs), and Virtual Lanes (VLs).

## Isolation Mechanisms

### Partition Keys (PKeys)

PKeys are the primary isolation mechanism in IB networks:
- Each port can be a member of multiple partitions
- Traffic between different partitions is blocked at the hardware level
- Similar to VLANs in Ethernet networks

```
Partition 0x7FFF (default) — all nodes
Partition 0x0001 (compute)  — GPU compute nodes
Partition 0x0002 (storage)  — storage traffic
Partition 0x0003 (mgmt)     — management traffic
```

### Service Levels (SLs)

SLs provide priority differentiation within a partition:
- 16 available SLs (0-15) in InfiniBand
- Mapped to Virtual Lanes by the subnet manager
- Higher SLs can be given priority scheduling

### Virtual Lanes (VLs)

VLs are the physical queues in IB switch ports:
- Typically 2-8 data VLs available per port (+ VL15 for management)
- SL-to-VL mapping determines which queue a packet enters
- Each VL has independent flow control and scheduling

## Configuration

### Creating Partitions (OpenSM)

Edit `/etc/opensm/partitions.conf`:
```
# partition_name, PKey, ipoib, flags: node_guids
Default=0x7FFF, ipoib: ALL=full
Compute=0x0001, ipoib: 0x0002c903000001, 0x0002c903000002=full
Storage=0x0002, ipoib: 0x0002c903000010, 0x0002c903000011=full
```

### SL-to-VL Mapping

Configure in OpenSM QoS policy:
```
# SL 0-3 -> VL 0 (best effort)
# SL 4-5 -> VL 1 (compute)
# SL 6-7 -> VL 2 (storage)
qos-sl2vl 0,0,0,0,1,1,2,2,0,0,0,0,0,0,0,0
```

## Verification

```bash
# List partitions on a port
smpquery pkeytable <lid> <port>

# Check SL-to-VL mapping on a switch
smpquery sl2vl <lid> <in_port> <out_port>

# Verify traffic isolation with ibdiagnet
ibdiagnet --pkey

# Test connectivity within a partition
ibping -P <pkey> <remote_lid>
```

## Best Practices

1. **Separate storage and compute**: Storage I/O bursts can disrupt latency-sensitive MPI traffic
2. **Dedicate a VL for management**: Ensures SM communication even under full congestion
3. **Use minimal partitions**: Excessive partitions complicate debugging
4. **Monitor per-VL counters**: Detect imbalances early with `perfquery` VL data counters
5. **Test failover paths**: Ensure isolated partitions still have redundant routes

## Impact on Performance

| Scenario | Without Isolation | With Isolation |
|----------|-------------------|----------------|
| MPI latency during I/O storm | 15-50 us | 2-3 us |
| GPU-to-GPU bandwidth under mixed load | 60% peak | 95% peak |
| Storage throughput consistency | Variable | Predictable |
