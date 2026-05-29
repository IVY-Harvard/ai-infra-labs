#!/bin/bash
# qos_config.sh - QoS Configuration for InfiniBand Subnet Manager
# Configures Service Level policies and SL-to-VL mapping for traffic prioritization

set -euo pipefail

OPENSM_CONF="/etc/opensm/opensm.conf"
QOS_POLICY="/etc/opensm/qos-policy.conf"

echo "=== InfiniBand QoS Configuration ==="
echo "Timestamp: $(date)"

# Check for root privileges
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script requires root privileges."
    exit 1
fi

# Backup existing configuration
if [ -f "$OPENSM_CONF" ]; then
    cp "$OPENSM_CONF" "${OPENSM_CONF}.bak.$(date +%Y%m%d%H%M%S)"
    echo "Backed up opensm.conf"
fi

# Enable QoS in OpenSM
echo "Enabling QoS in OpenSM configuration..."
if grep -q "^qos " "$OPENSM_CONF" 2>/dev/null; then
    sed -i 's/^qos .*/qos TRUE/' "$OPENSM_CONF"
else
    echo "qos TRUE" >> "$OPENSM_CONF"
fi

# Write QoS policy file
echo "Writing QoS policy..."
cat > "$QOS_POLICY" << 'EOF'
# QoS Policy Configuration
# SL 0: Best-effort (default)
# SL 1: Bulk data transfer
# SL 4: Compute / MPI (high priority)
# SL 6: Storage I/O
# SL 7: Management (highest priority)

qos-sl2vl 0,0,0,0,1,1,2,2,3,0,0,0,0,0,0,3

# High-priority weight for compute VL
qos-vl-high-limit 4

# Rate limiting for bulk transfers
qos-sl-rate-limit SL1=80%
EOF

echo "QoS policy written to $QOS_POLICY"

# Restart OpenSM to apply changes
echo "Restarting OpenSM..."
if systemctl is-active --quiet opensm; then
    systemctl restart opensm
    echo "OpenSM restarted successfully."
else
    echo "WARNING: OpenSM service not found. Apply config manually."
fi

echo ""
echo "QoS configuration complete."
echo "Verify with: smpquery sl2vl <switch_lid> <port> <port>"
