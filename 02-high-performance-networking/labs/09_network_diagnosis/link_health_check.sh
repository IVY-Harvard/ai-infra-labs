#!/bin/bash
# link_health_check.sh - InfiniBand Link Health Check Script
# Checks link state, port speed, and error counters for all IB ports

set -euo pipefail

LOG_FILE="/tmp/ib_link_health_$(date +%Y%m%d_%H%M%S).log"

echo "=== InfiniBand Link Health Check ===" | tee "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Hostname: $(hostname)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check if ibstat is available
if ! command -v ibstat &>/dev/null; then
    echo "ERROR: ibstat not found. Install infiniband-diags." | tee -a "$LOG_FILE"
    exit 1
fi

# Get all IB devices
DEVICES=$(ibstat -l 2>/dev/null)
if [ -z "$DEVICES" ]; then
    echo "WARNING: No InfiniBand devices found." | tee -a "$LOG_FILE"
    exit 1
fi

for DEV in $DEVICES; do
    echo "--- Device: $DEV ---" | tee -a "$LOG_FILE"
    PORTS=$(ibstat "$DEV" | grep -c "Port " || true)
    for PORT in $(seq 1 "$PORTS"); do
        echo "  Port $PORT:" | tee -a "$LOG_FILE"
        STATE=$(ibstat "$DEV" "$PORT" | grep "State:" | awk '{print $2}')
        SPEED=$(ibstat "$DEV" "$PORT" | grep "Rate:" | awk '{print $2}')
        echo "    State: $STATE" | tee -a "$LOG_FILE"
        echo "    Speed: ${SPEED} Gb/s" | tee -a "$LOG_FILE"

        if [ "$STATE" != "Active" ]; then
            echo "    [ALERT] Port is NOT active!" | tee -a "$LOG_FILE"
        fi

        # Check error counters via perfquery
        if command -v perfquery &>/dev/null; then
            ERRORS=$(perfquery -x "$PORT" 2>/dev/null | grep -E "Err|Drop" || true)
            if [ -n "$ERRORS" ]; then
                echo "    Error Counters:" | tee -a "$LOG_FILE"
                echo "$ERRORS" | sed 's/^/      /' | tee -a "$LOG_FILE"
            fi
        fi
    done
done

echo "" | tee -a "$LOG_FILE"
echo "Health check complete. Log saved to: $LOG_FILE"
