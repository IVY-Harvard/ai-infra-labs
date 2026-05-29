# Adaptive Routing in InfiniBand Fabrics

## Concept

Adaptive routing dynamically selects paths through the network fabric based on current link utilization, avoiding congested routes and distributing traffic more evenly across available paths.

## Why Adaptive Routing?

In static (deterministic) routing, each source-destination pair uses a single fixed path. This leads to:
- **Hot spots** when multiple flows share a link
- **Underutilization** of alternative paths
- **Performance degradation** under congestion

Adaptive routing solves these problems by allowing switches to forward packets along less-loaded paths.

## Fat-Tree Topology Context

In a fat-tree (Clos) topology, multiple equal-cost paths exist between any two endpoints. Adaptive routing exploits this structural redundancy:

```
        [Core Switches]
       /    |    |    \
  [Agg1] [Agg2] [Agg3] [Agg4]
   / \    / \    / \    / \
 [Leaf switches — connected to compute nodes]
```

At each upward hop, the switch can choose among several uplinks. Adaptive routing selects the least-loaded one.

## Mechanisms

### Port-Level Load Balancing
- Switch monitors egress queue depth on each candidate port
- Selects port with shortest queue for each packet/flow

### Congestion-Aware Routing
- Subnet manager collects congestion notifications (FECN/BECN)
- Routing tables updated to avoid persistently congested links

### Per-Packet vs. Per-Flow
| Mode | Pros | Cons |
|------|------|------|
| Per-packet | Maximum load distribution | Possible reordering |
| Per-flow | No reordering | Less granular balancing |

## Configuration (OpenSM)

Enable adaptive routing in OpenSM configuration:

```bash
# /etc/opensm/opensm.conf
routing_engine ar_updn
ar_mode 1                    # 1 = adaptive routing enabled
ar_threshold 50              # Queue depth threshold (%)
```

Restart OpenSM after changes:
```bash
systemctl restart opensm
```

## Verification

Check routing distribution:
```bash
# View LID routes on a switch
smpquery portinfo <switch_lid> <port>

# Monitor per-port traffic with perfquery
perfquery -x <lid> <port>

# Verify multiple paths are being used
ibtracert <src_lid> <dst_lid>
```

## Performance Impact

Typical improvements with adaptive routing enabled:
- Single-flow: minimal change (still uses one path)
- Multi-flow (all-to-all): 20-40% bandwidth improvement
- Bisection bandwidth: approaches theoretical maximum
