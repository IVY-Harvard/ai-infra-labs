#!/usr/bin/env bash
###############################################################################
# pfc_ecn_config.sh - PFC 和 ECN 综合配置脚本
#
# 功能: 配置无损以太网 (Lossless Ethernet) 所需的 PFC 和 ECN，包括：
#   - PFC (Priority Flow Control, 802.1Qbb) 配置
#   - ECN (Explicit Congestion Notification) 配置
#   - DCQCN (Data Center QCN) 综合配置
#   - 配置验证与状态查看
#
# 用法: sudo ./pfc_ecn_config.sh --interface <iface> --mode <pfc|ecn|full> [options]
#
# 前提条件:
#   - NVIDIA/Mellanox 网卡 (ConnectX-4+)
#   - mlnx_qos 工具 (MLNX_OFED)
#   - root 权限
###############################################################################

set -euo pipefail

# ─── 颜色 ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ─── 默认参数 ────────────────────────────────────────────────────────────────
INTERFACE=""
MODE=""                    # pfc, ecn, full
PRIORITY=3                 # RoCE 使用的优先级 (0-7)
DSCP=26                    # AF31, 常见 RoCE DSCP 值
BUFFER_SIZE=312500         # PFC 缓冲区阈值 (bytes), 约 2.5Mb
ECN_MIN_THRESHOLD=150000   # ECN 标记最小阈值 (bytes)
ECN_MAX_THRESHOLD=1500000  # ECN 标记最大阈值 (bytes)
ECN_PROBABILITY=100        # ECN 标记概率 (百分比, 到达 max 时)
CNP_DSCP=48                # CNP (拥塞通知包) DSCP 值
CNP_PRIORITY=6             # CNP 优先级
SHOW_ONLY=false

# ─── 参数解析 ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --interface|-i)
            INTERFACE="$2"
            shift 2
            ;;
        --mode|-m)
            MODE="$2"
            shift 2
            ;;
        --priority|-p)
            PRIORITY="$2"
            shift 2
            ;;
        --dscp|-d)
            DSCP="$2"
            shift 2
            ;;
        --buffer-size|-b)
            BUFFER_SIZE="$2"
            shift 2
            ;;
        --ecn-min)
            ECN_MIN_THRESHOLD="$2"
            shift 2
            ;;
        --ecn-max)
            ECN_MAX_THRESHOLD="$2"
            shift 2
            ;;
        --show|-s)
            SHOW_ONLY=true
            shift
            ;;
        --help|-h)
            cat << 'HELP'
Usage: sudo ./pfc_ecn_config.sh --interface <iface> --mode <pfc|ecn|full> [options]

Modes:
  pfc    仅配置 PFC (Priority Flow Control)
  ecn    仅配置 ECN (Explicit Congestion Notification)
  full   完整 DCQCN 配置 (PFC + ECN + CNP)

Options:
  --interface, -i <iface>   网络接口名称 (必选)
  --mode, -m <mode>         配置模式: pfc / ecn / full (必选)
  --priority, -p <0-7>      RoCE 流量优先级 (默认: 3)
  --dscp, -d <value>        DSCP 值 (默认: 26, AF31)
  --buffer-size, -b <bytes> PFC 缓冲区大小 (默认: 312500)
  --ecn-min <bytes>         ECN 最小标记阈值 (默认: 150000)
  --ecn-max <bytes>         ECN 最大标记阈值 (默认: 1500000)
  --show, -s                仅显示当前配置状态
  --help, -h                显示帮助

Examples:
  # 仅配置 PFC on priority 3
  sudo ./pfc_ecn_config.sh -i ens1f0 -m pfc -p 3

  # 仅配置 ECN
  sudo ./pfc_ecn_config.sh -i ens1f0 -m ecn -p 3

  # 完整 DCQCN (PFC + ECN + CNP)
  sudo ./pfc_ecn_config.sh -i ens1f0 -m full -p 3

  # 查看当前状态
  sudo ./pfc_ecn_config.sh -i ens1f0 --show
HELP
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1
            ;;
    esac
done

# 参数校验
if [[ -z "$INTERFACE" ]]; then
    echo -e "${RED}错误: 必须指定 --interface${NC}"
    exit 1
fi

if [[ "$SHOW_ONLY" == false ]] && [[ -z "$MODE" ]]; then
    echo -e "${RED}错误: 必须指定 --mode (pfc|ecn|full) 或 --show${NC}"
    exit 1
fi

# ─── 工具函数 ────────────────────────────────────────────────────────────────

header() {
    echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} ${BOLD}$1${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
}

section() {
    echo -e "\n${CYAN}▶ $1${NC}"
}

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
err()  { echo -e "  ${RED}✗${NC} $1"; }
info() { echo -e "  ${BLUE}ℹ${NC} $1"; }

cmd_exists() { command -v "$1" &>/dev/null; }

# ─── 前置检查 ────────────────────────────────────────────────────────────────

check_prerequisites() {
    section "前置检查"

    if [[ $EUID -ne 0 ]]; then
        err "需要 root 权限"
        exit 1
    fi
    ok "root 权限确认"

    if [[ ! -d "/sys/class/net/$INTERFACE" ]]; then
        err "接口 $INTERFACE 不存在"
        exit 1
    fi
    ok "接口 $INTERFACE 存在"

    if ! cmd_exists mlnx_qos; then
        err "mlnx_qos 不可用 (需要 MLNX_OFED)"
        info "安装: yum install mlnx-tools 或 apt-get install mlnx-tools"
        exit 1
    fi
    ok "mlnx_qos 可用"

    if cmd_exists tc; then
        ok "tc (iproute2) 可用"
    else
        warn "tc 不可用，部分功能受限"
    fi

    # 检查 lldptool (用于 DCBX)
    if cmd_exists lldptool; then
        ok "lldptool 可用 (DCBX 支持)"
    else
        warn "lldptool 不可用 (DCBX 自动协商不可用)"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# 显示当前状态
# ═══════════════════════════════════════════════════════════════════════════════

show_status() {
    header "当前 QoS/PFC/ECN 配置状态 — $INTERFACE"

    section "1. mlnx_qos 完整状态"
    mlnx_qos -i "$INTERFACE" 2>&1 || warn "mlnx_qos 查询失败"

    section "2. PFC 统计"
    if [[ -d "/sys/class/net/$INTERFACE/statistics" ]]; then
        echo "  接口统计:"
        for f in /sys/class/net/$INTERFACE/statistics/rx_*; do
            val=$(cat "$f" 2>/dev/null || echo 0)
            [[ "$val" != "0" ]] && echo "    $(basename $f): $val"
        done | head -10
    fi

    # ethtool PFC 统计
    if cmd_exists ethtool; then
        echo ""
        info "PFC 帧统计 (ethtool):"
        ethtool -S "$INTERFACE" 2>/dev/null | grep -i "pfc\|pause" | head -20 | sed 's/^/    /' || true
    fi

    section "3. ECN 配置 (sysfs)"
    ECN_DIR="/sys/class/net/$INTERFACE/ecn"
    if [[ -d "$ECN_DIR" ]]; then
        info "ECN sysfs 配置:"
        find "$ECN_DIR" -type f -exec sh -c 'echo "  $(basename {}): $(cat {})"' \; 2>/dev/null || true
    else
        info "ECN sysfs 目录不存在 (可能通过其他方式配置)"
    fi

    # 检查内核 ECN
    section "4. 内核 ECN 设置"
    tcp_ecn=$(sysctl -n net.ipv4.tcp_ecn 2>/dev/null || echo "N/A")
    info "net.ipv4.tcp_ecn = $tcp_ecn"
    echo "    (0=禁用, 1=启用, 2=服务端响应)"

    section "5. TC Qdisc 配置"
    if cmd_exists tc; then
        tc qdisc show dev "$INTERFACE" 2>/dev/null | sed 's/^/    /' || true
    fi

    section "6. DCBX 状态"
    if cmd_exists lldptool; then
        info "DCBX 操作模式:"
        lldptool -ti "$INTERFACE" -V PFC 2>/dev/null | sed 's/^/    /' || true
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# PFC 配置
# ═══════════════════════════════════════════════════════════════════════════════

configure_pfc() {
    header "配置 PFC (Priority Flow Control)"
    echo ""
    info "目标: 在 Priority $PRIORITY 上启用 PFC (无损传输)"
    info "原理: 当缓冲区达到阈值时，发送 PAUSE 帧暂停该优先级的流量"
    echo ""

    # 步骤 1: 设置 Trust Mode
    section "步骤 1: 设置 Trust Mode 为 DSCP"
    mlnx_qos -i "$INTERFACE" --trust=dscp 2>&1 && \
        ok "Trust mode = DSCP" || \
        warn "Trust mode 设置失败"

    # 步骤 2: DSCP-to-Priority 映射
    section "步骤 2: DSCP $DSCP → Priority $PRIORITY 映射"
    mlnx_qos -i "$INTERFACE" --dscp2prio "set,$DSCP,$PRIORITY" 2>&1 && \
        ok "DSCP $DSCP → Priority $PRIORITY" || \
        warn "DSCP 映射设置失败"

    # 步骤 3: Priority-to-TC 映射
    section "步骤 3: Priority → Traffic Class 映射"
    # 构建 prio_tc 字符串: 8 个 TC 值，对应 priority 0-7
    PRIO_TC=""
    for i in $(seq 0 7); do
        if [[ $i -eq $PRIORITY ]]; then
            PRIO_TC="${PRIO_TC}${PRIORITY}"
        else
            PRIO_TC="${PRIO_TC}0"
        fi
        [[ $i -lt 7 ]] && PRIO_TC="${PRIO_TC},"
    done

    mlnx_qos -i "$INTERFACE" --prio_tc="$PRIO_TC" 2>&1 && \
        ok "Priority-to-TC: $PRIO_TC" || \
        warn "Priority-to-TC 映射设置失败"

    # 步骤 4: 启用 PFC on specified priority
    section "步骤 4: 启用 PFC (Priority $PRIORITY)"
    # 构建 PFC 启用字符串
    PFC_EN=""
    for i in $(seq 0 7); do
        if [[ $i -eq $PRIORITY ]]; then
            PFC_EN="${PFC_EN}1"
        else
            PFC_EN="${PFC_EN}0"
        fi
        [[ $i -lt 7 ]] && PFC_EN="${PFC_EN},"
    done

    mlnx_qos -i "$INTERFACE" --pfc="$PFC_EN" 2>&1 && \
        ok "PFC 已启用: $PFC_EN" || \
        warn "PFC 启用失败"

    # 步骤 5: 配置缓冲区大小
    section "步骤 5: 配置 PFC 缓冲区"
    # 构建 buffer size 字符串
    BUFFER_STR=""
    for i in $(seq 0 7); do
        if [[ $i -eq $PRIORITY ]]; then
            BUFFER_STR="${BUFFER_STR}${BUFFER_SIZE}"
        else
            BUFFER_STR="${BUFFER_STR}0"
        fi
        [[ $i -lt 7 ]] && BUFFER_STR="${BUFFER_STR},"
    done

    mlnx_qos -i "$INTERFACE" --buffer_size="$BUFFER_STR" 2>&1 && \
        ok "缓冲区配置: TC$PRIORITY = ${BUFFER_SIZE} bytes" || \
        warn "缓冲区设置失败 (某些固件版本不支持直接设置)"

    # 步骤 6: 禁用全局 PAUSE (仅使用 PFC)
    section "步骤 6: 禁用全局 PAUSE 帧"
    if cmd_exists ethtool; then
        ethtool -A "$INTERFACE" rx off tx off 2>&1 && \
            ok "全局 PAUSE 已禁用 (使用 PFC 代替)" || \
            warn "全局 PAUSE 禁用失败"
    fi

    ok "PFC 配置完成"
}

# ═══════════════════════════════════════════════════════════════════════════════
# ECN 配置
# ═══════════════════════════════════════════════════════════════════════════════

configure_ecn() {
    header "配置 ECN (Explicit Congestion Notification)"
    echo ""
    info "目标: 启用 ECN 标记，实现端到端拥塞感知"
    info "原理: 交换机在缓冲区接近满时标记 ECN CE 位"
    info "      接收端生成 CNP (Congestion Notification Packet)"
    info "      发送端收到 CNP 后降低发送速率"
    echo ""

    # 步骤 1: 内核 ECN 支持
    section "步骤 1: 启用内核 ECN 支持"
    sysctl -w net.ipv4.tcp_ecn=1 >/dev/null 2>&1 && \
        ok "net.ipv4.tcp_ecn = 1 (已启用)" || \
        warn "内核 ECN 设置失败"

    # 步骤 2: 使用 mlnx_qos 设置 ECN on TC
    section "步骤 2: 在 TC $PRIORITY 上启用 ECN"

    # 构建 ECN 启用字符串
    ECN_EN=""
    for i in $(seq 0 7); do
        if [[ $i -eq $PRIORITY ]]; then
            ECN_EN="${ECN_EN}1"
        else
            ECN_EN="${ECN_EN}0"
        fi
        [[ $i -lt 7 ]] && ECN_EN="${ECN_EN},"
    done

    mlnx_qos -i "$INTERFACE" --ecn="$ECN_EN" 2>&1 && \
        ok "ECN 已启用: $ECN_EN" || \
        warn "ECN 启用失败 (尝试替代方法)"

    # 步骤 3: 配置 ECN 标记阈值
    section "步骤 3: 配置 ECN 标记阈值"
    info "最小阈值: $ECN_MIN_THRESHOLD bytes (开始概率标记)"
    info "最大阈值: $ECN_MAX_THRESHOLD bytes (100% 标记)"

    # 通过 sysfs 设置阈值 (Mellanox 特定)
    ECN_SYSFS_BASE="/sys/class/net/$INTERFACE/ecn/roce_np"
    if [[ -d "$ECN_SYSFS_BASE" ]]; then
        # 设置 min threshold
        if [[ -f "$ECN_SYSFS_BASE/min_time_between_cnps" ]]; then
            echo 4 > "$ECN_SYSFS_BASE/min_time_between_cnps" 2>/dev/null && \
                ok "CNP 最小间隔: 4 usec" || true
        fi
    fi

    # 使用 tc-red 配置 ECN (通用方法)
    if cmd_exists tc; then
        section "步骤 3b: tc-red ECN 配置 (通用方法)"

        # 删除已有 qdisc
        tc qdisc del dev "$INTERFACE" root 2>/dev/null || true

        # 创建 PRIO qdisc 和 RED with ECN
        tc qdisc add dev "$INTERFACE" root handle 1: prio bands 8 2>/dev/null || true

        # 在指定 band 上添加 RED with ECN
        tc qdisc add dev "$INTERFACE" parent "1:$((PRIORITY+1))" handle "$((PRIORITY+1))0:" \
            red limit $((ECN_MAX_THRESHOLD * 3)) \
            min "$ECN_MIN_THRESHOLD" \
            max "$ECN_MAX_THRESHOLD" \
            probability 0.${ECN_PROBABILITY} \
            ecn 2>&1 && \
            ok "tc RED+ECN 已配置 (band $((PRIORITY+1)))" || \
            warn "tc RED+ECN 配置失败"
    fi

    # 步骤 4: DCQCN 参数 (Mellanox 特定)
    section "步骤 4: DCQCN 算法参数"

    DCQCN_SYSFS="/sys/class/net/$INTERFACE/ecn/roce_rp"
    if [[ -d "$DCQCN_SYSFS" ]]; then
        info "配置 DCQCN 反应点 (RP) 参数:"

        # 初始速率降低因子
        [[ -f "$DCQCN_SYSFS/rate_reduce_monitor_period" ]] && \
            echo 4 > "$DCQCN_SYSFS/rate_reduce_monitor_period" 2>/dev/null && \
            ok "rate_reduce_monitor_period = 4 usec" || true

        # 速率恢复相关
        [[ -f "$DCQCN_SYSFS/initial_alpha_value" ]] && \
            echo 1023 > "$DCQCN_SYSFS/initial_alpha_value" 2>/dev/null && \
            ok "initial_alpha_value = 1023" || true

        [[ -f "$DCQCN_SYSFS/rpg_time_reset" ]] && \
            echo 300 > "$DCQCN_SYSFS/rpg_time_reset" 2>/dev/null && \
            ok "rpg_time_reset = 300 usec" || true

        [[ -f "$DCQCN_SYSFS/rpg_hai_rate" ]] && \
            echo 5 > "$DCQCN_SYSFS/rpg_hai_rate" 2>/dev/null && \
            ok "rpg_hai_rate = 5 (Hyper-Active Increase)" || true

        [[ -f "$DCQCN_SYSFS/rpg_threshold" ]] && \
            echo 1 > "$DCQCN_SYSFS/rpg_threshold" 2>/dev/null && \
            ok "rpg_threshold = 1" || true
    else
        info "DCQCN sysfs 接口不存在 (可能使用不同版本的驱动)"
        info "参考: /sys/class/net/<iface>/ecn/roce_rp/"
    fi

    ok "ECN 配置完成"
}

# ═══════════════════════════════════════════════════════════════════════════════
# CNP 配置 (DCQCN 完整配置的一部分)
# ═══════════════════════════════════════════════════════════════════════════════

configure_cnp() {
    header "配置 CNP (Congestion Notification Packets)"
    echo ""
    info "CNP 是接收端发回发送端的拥塞通知报文"
    info "需要为 CNP 配置独立的 DSCP/Priority，避免与数据流量竞争"
    echo ""

    section "CNP DSCP 配置"
    info "CNP DSCP: $CNP_DSCP (与数据 DSCP $DSCP 不同)"
    info "CNP Priority: $CNP_PRIORITY"

    # CNP DSCP-to-Priority 映射
    if cmd_exists mlnx_qos; then
        mlnx_qos -i "$INTERFACE" --dscp2prio "set,$CNP_DSCP,$CNP_PRIORITY" 2>&1 && \
            ok "CNP DSCP $CNP_DSCP → Priority $CNP_PRIORITY" || \
            warn "CNP DSCP 映射失败"
    fi

    # 通过 sysfs 设置 CNP DSCP
    CNP_DSCP_SYSFS="/sys/class/net/$INTERFACE/ecn/roce_np/cnp_dscp"
    if [[ -f "$CNP_DSCP_SYSFS" ]]; then
        echo "$CNP_DSCP" > "$CNP_DSCP_SYSFS" 2>/dev/null && \
            ok "CNP DSCP 设置为 $CNP_DSCP (via sysfs)" || \
            warn "CNP DSCP sysfs 设置失败"
    fi

    CNP_PRIO_SYSFS="/sys/class/net/$INTERFACE/ecn/roce_np/cnp_802p_prio"
    if [[ -f "$CNP_PRIO_SYSFS" ]]; then
        echo "$CNP_PRIORITY" > "$CNP_PRIO_SYSFS" 2>/dev/null && \
            ok "CNP 802.1p Priority 设置为 $CNP_PRIORITY (via sysfs)" || \
            warn "CNP Priority sysfs 设置失败"
    fi

    ok "CNP 配置完成"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════════════════════════

check_prerequisites

if [[ "$SHOW_ONLY" == true ]]; then
    show_status
    exit 0
fi

echo ""
info "配置模式: $MODE"
info "接口: $INTERFACE"
info "RoCE Priority: $PRIORITY"
info "DSCP: $DSCP"
echo ""

case "$MODE" in
    pfc)
        configure_pfc
        ;;
    ecn)
        configure_ecn
        ;;
    full)
        configure_pfc
        configure_ecn
        configure_cnp
        ;;
    *)
        err "未知模式: $MODE (支持: pfc, ecn, full)"
        exit 1
        ;;
esac

# ─── 最终验证 ────────────────────────────────────────────────────────────────

header "最终配置验证"

section "mlnx_qos 当前状态"
mlnx_qos -i "$INTERFACE" 2>&1 | head -40 || true

section "PFC 帧计数器"
if cmd_exists ethtool; then
    ethtool -S "$INTERFACE" 2>/dev/null | grep -i "pfc" | head -10 | sed 's/^/    /' || true
fi

# ─── 配置持久化建议 ──────────────────────────────────────────────────────────

header "配置持久化"
echo ""
info "当前配置在系统重启后会丢失。持久化方法："
echo ""
echo "  方法 1: 添加到 /etc/rc.local 或 systemd service"
echo ""
echo "  方法 2: 使用 Netplan/NetworkManager dispatcher"
echo ""
echo "  方法 3: 创建 udev 规则"
echo "  示例:"
echo "    # /etc/udev/rules.d/99-roce-qos.rules"
echo "    ACTION==\"add\", SUBSYSTEM==\"net\", NAME==\"$INTERFACE\", \\"
echo "      RUN+=\"/usr/local/bin/roce_qos_setup.sh\""
echo ""
echo "  方法 4: MLNX_OFED mlnx_tune 工具"
echo "    mlnx_tune -p THROUGHPUT"
echo ""

# ─── 交换机配置提醒 ──────────────────────────────────────────────────────────

header "交换机配置提醒"
echo ""
warn "PFC/ECN 需要主机和交换机两端配置！"
echo ""
echo "  交换机需要配置:"
echo "    1. PFC: 在对应端口和 Priority 上启用 PFC"
echo "    2. ECN: 配置 WRED/ECN 标记阈值"
echo "    3. Buffer: 为无损 TC 分配足够缓冲区"
echo "    4. DCBX: 如果使用自动协商"
echo ""
echo "  参考交换机配置:"
echo "    NVIDIA Spectrum: nv set qos pfc priority $PRIORITY"
echo "    Cisco Nexus:     priority-flow-control priority $PRIORITY no-drop"
echo "    Arista EOS:      priority-flow-control priority $PRIORITY no-drop"
echo ""
ok "配置脚本执行完毕"
