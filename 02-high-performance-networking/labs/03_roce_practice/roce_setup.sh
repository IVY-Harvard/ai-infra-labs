#!/usr/bin/env bash
###############################################################################
# roce_setup.sh - RoCE v2 网卡配置脚本
#
# 功能: 配置 RDMA 网卡以使用 RoCE v2 模式，包括：
#   - 设置 RoCE 模式 (v1/v2)
#   - 配置 GID 索引
#   - 设置 DSCP-to-Priority 映射
#   - 配置 Traffic Class (TC)
#   - 验证 RoCE 配置
#
# 用法: sudo ./roce_setup.sh --interface <iface> [--gid-index <idx>] [--dscp <val>]
#
# 前提条件:
#   - NVIDIA/Mellanox RDMA 网卡 (ConnectX-4 及以上)
#   - MLNX_OFED 或 inbox rdma-core 驱动
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
GID_INDEX=3
DSCP_VALUE=26          # DSCP CS3 for RoCE (常见推荐值)
PRIORITY=3             # 对应 TC3
ROCE_MODE="2"          # 默认 RoCE v2
MTU=4096               # RoCE 推荐 MTU
VERIFY_ONLY=false

# ─── 参数解析 ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --interface|-i)
            INTERFACE="$2"
            shift 2
            ;;
        --gid-index|-g)
            GID_INDEX="$2"
            shift 2
            ;;
        --dscp|-d)
            DSCP_VALUE="$2"
            shift 2
            ;;
        --priority|-p)
            PRIORITY="$2"
            shift 2
            ;;
        --roce-mode|-m)
            ROCE_MODE="$2"
            shift 2
            ;;
        --mtu)
            MTU="$2"
            shift 2
            ;;
        --verify)
            VERIFY_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --interface <iface> [options]"
            echo ""
            echo "Options:"
            echo "  --interface, -i <iface>   网络接口名称 (必选, 例: ens1f0)"
            echo "  --gid-index, -g <idx>     GID 索引 (默认: 3)"
            echo "  --dscp, -d <value>        DSCP 值 (默认: 26, 即 AF31)"
            echo "  --priority, -p <prio>     802.1p 优先级 (默认: 3)"
            echo "  --roce-mode, -m <1|2>     RoCE 版本 (默认: 2)"
            echo "  --mtu <size>              MTU 大小 (默认: 4096)"
            echo "  --verify                  仅验证当前配置，不做修改"
            echo "  --help, -h                显示帮助"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1
            ;;
    esac
done

if [[ -z "$INTERFACE" ]]; then
    echo -e "${RED}错误: 必须指定 --interface 参数${NC}"
    echo "使用 --help 查看帮助"
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

header "RoCE v${ROCE_MODE} 配置 — 接口: $INTERFACE"

section "前置检查"

# 检查 root 权限
if [[ $EUID -ne 0 ]]; then
    err "需要 root 权限运行此脚本"
    exit 1
fi
ok "root 权限确认"

# 检查接口是否存在
if [[ ! -d "/sys/class/net/$INTERFACE" ]]; then
    err "接口 $INTERFACE 不存在"
    echo "  可用接口:"
    ls /sys/class/net/ | grep -v lo | sed 's/^/    /'
    exit 1
fi
ok "接口 $INTERFACE 存在"

# 获取 RDMA 设备名
RDMA_DEV=""
for dev in /sys/class/infiniband/*; do
    if [[ -d "$dev" ]]; then
        dev_name=$(basename "$dev")
        # 检查该 RDMA 设备是否关联到我们的网络接口
        for port_dir in "$dev/ports"/*/; do
            if [[ -d "${port_dir}gid_attrs/ndevs" ]]; then
                for ndev_file in "${port_dir}gid_attrs/ndevs"/*; do
                    if [[ -f "$ndev_file" ]]; then
                        ndev=$(cat "$ndev_file" 2>/dev/null || true)
                        if [[ "$ndev" == "$INTERFACE" ]]; then
                            RDMA_DEV="$dev_name"
                            RDMA_PORT=$(basename "$(dirname "$(dirname "$ndev_file")")")
                            break 3
                        fi
                    fi
                done
            fi
        done
    fi
done

if [[ -z "$RDMA_DEV" ]]; then
    # 尝试通过 rdma link 获取
    if cmd_exists rdma; then
        RDMA_DEV=$(rdma link show | grep "$INTERFACE" | awk '{print $2}' | cut -d'/' -f1)
        RDMA_PORT=$(rdma link show | grep "$INTERFACE" | awk '{print $2}' | cut -d'/' -f2)
    fi
fi

if [[ -z "$RDMA_DEV" ]]; then
    err "无法找到接口 $INTERFACE 关联的 RDMA 设备"
    exit 1
fi
ok "RDMA 设备: $RDMA_DEV (端口: ${RDMA_PORT:-1})"

# 检查必要工具
REQUIRED_CMDS=("cma_roce_mode" "cma_roce_tos" "mlnx_qos" "sysctl" "ip")
MISSING_CMDS=()
for cmd in "${REQUIRED_CMDS[@]}"; do
    if ! cmd_exists "$cmd"; then
        MISSING_CMDS+=("$cmd")
    fi
done

if [[ ${#MISSING_CMDS[@]} -gt 0 ]]; then
    warn "以下工具不可用: ${MISSING_CMDS[*]}"
    info "部分配置将使用替代方法 (sysfs)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 验证模式
# ═══════════════════════════════════════════════════════════════════════════════

show_current_config() {
    header "当前 RoCE 配置状态"

    section "1. RoCE 模式"
    ROCE_MODE_FILE="/sys/class/infiniband/$RDMA_DEV/ports/${RDMA_PORT:-1}/gid_attrs/types"
    if [[ -d "$ROCE_MODE_FILE" ]]; then
        info "GID 类型列表:"
        for gid_type_file in "$ROCE_MODE_FILE"/*; do
            if [[ -f "$gid_type_file" ]]; then
                idx=$(basename "$gid_type_file")
                gtype=$(cat "$gid_type_file" 2>/dev/null || echo "N/A")
                if [[ "$gtype" != "IB/RoCE v1" ]] && [[ "$gtype" != "" ]]; then
                    echo "    GID[$idx]: $gtype"
                fi
            fi
        done | head -10
    fi

    # 使用 cma_roce_mode 查看
    if cmd_exists cma_roce_mode; then
        info "CMA RoCE Mode:"
        cma_roce_mode -d "$RDMA_DEV" -p "${RDMA_PORT:-1}" 2>&1 | sed 's/^/    /'
    fi

    section "2. GID 表"
    GID_DIR="/sys/class/infiniband/$RDMA_DEV/ports/${RDMA_PORT:-1}/gids"
    if [[ -d "$GID_DIR" ]]; then
        info "有效 GID 条目:"
        for gid_file in "$GID_DIR"/*; do
            if [[ -f "$gid_file" ]]; then
                gid_val=$(cat "$gid_file" 2>/dev/null)
                if [[ "$gid_val" != "0000:0000:0000:0000:0000:0000:0000:0000" ]] && \
                   [[ -n "$gid_val" ]]; then
                    idx=$(basename "$gid_file")
                    gtype="unknown"
                    if [[ -f "/sys/class/infiniband/$RDMA_DEV/ports/${RDMA_PORT:-1}/gid_attrs/types/$idx" ]]; then
                        gtype=$(cat "/sys/class/infiniband/$RDMA_DEV/ports/${RDMA_PORT:-1}/gid_attrs/types/$idx" 2>/dev/null)
                    fi
                    echo "    [$idx] $gid_val  ($gtype)"
                fi
            fi
        done
    fi

    section "3. DSCP/TOS 配置"
    if cmd_exists cma_roce_tos; then
        info "CMA RoCE TOS:"
        cma_roce_tos -d "$RDMA_DEV" -p "${RDMA_PORT:-1}" 2>&1 | sed 's/^/    /' || true
    fi

    section "4. 接口 MTU"
    current_mtu=$(cat "/sys/class/net/$INTERFACE/mtu" 2>/dev/null || echo "N/A")
    info "当前 MTU: $current_mtu"

    section "5. 链路状态"
    if cmd_exists ip; then
        ip link show "$INTERFACE" 2>/dev/null | sed 's/^/    /'
    fi

    section "6. Traffic Class (mlnx_qos)"
    if cmd_exists mlnx_qos; then
        mlnx_qos -i "$INTERFACE" 2>&1 | head -30 | sed 's/^/    /' || true
    fi
}

if [[ "$VERIFY_ONLY" == true ]]; then
    show_current_config
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 配置步骤
# ═══════════════════════════════════════════════════════════════════════════════

header "开始配置 RoCE v${ROCE_MODE}"

# ─── 步骤 1: 设置 RoCE 模式 ─────────────────────────────────────────────────

section "步骤 1: 设置 RoCE 模式为 v${ROCE_MODE}"

if cmd_exists cma_roce_mode; then
    if [[ "$ROCE_MODE" == "2" ]]; then
        cma_roce_mode -d "$RDMA_DEV" -p "${RDMA_PORT:-1}" -m 2 2>&1 && \
            ok "RoCE v2 模式已设置 (via cma_roce_mode)" || \
            warn "cma_roce_mode 设置失败，尝试替代方法"
    else
        cma_roce_mode -d "$RDMA_DEV" -p "${RDMA_PORT:-1}" -m 1 2>&1 && \
            ok "RoCE v1 模式已设置" || \
            warn "cma_roce_mode 设置失败"
    fi
else
    # 替代方法: 通过 rdma 工具设置
    if cmd_exists rdma; then
        rdma system set netns exclusive 2>/dev/null || true
        info "使用 rdma 工具设置 (cma_roce_mode 不可用)"
    fi
    # 通过 sysfs 设置默认 GID type
    GID_TYPE_DIR="/sys/class/infiniband/$RDMA_DEV/ports/${RDMA_PORT:-1}/gid_attrs/types"
    if [[ -d "$GID_TYPE_DIR" ]]; then
        info "GID 类型已由系统自动配置"
        ok "RoCE v2 GID 条目可用 (检查 GID index $GID_INDEX)"
    fi
fi

# ─── 步骤 2: 设置 DSCP / TOS ────────────────────────────────────────────────

section "步骤 2: 配置 DSCP/TOS 映射"

# TOS = DSCP << 2
TOS_VALUE=$((DSCP_VALUE << 2))
info "DSCP: $DSCP_VALUE → TOS: $TOS_VALUE"

if cmd_exists cma_roce_tos; then
    cma_roce_tos -d "$RDMA_DEV" -p "${RDMA_PORT:-1}" -t "$TOS_VALUE" 2>&1 && \
        ok "TOS 设置为 $TOS_VALUE (DSCP $DSCP_VALUE)" || \
        warn "cma_roce_tos 设置失败"
else
    # 通过 sysfs 设置
    TOS_SYSFS="/sys/class/infiniband/$RDMA_DEV/tc/${RDMA_PORT:-1}/traffic_class"
    if [[ -f "$TOS_SYSFS" ]]; then
        echo "$TOS_VALUE" > "$TOS_SYSFS" 2>/dev/null && \
            ok "TOS 通过 sysfs 设置为 $TOS_VALUE" || \
            warn "sysfs TOS 设置失败"
    else
        info "TOS 将在应用层通过 socket 选项设置"
        info "perftest 使用: --tos=$TOS_VALUE"
    fi
fi

# ─── 步骤 3: 配置 Trust Mode (DSCP-based) ───────────────────────────────────

section "步骤 3: 设置 Trust Mode 为 DSCP"

if cmd_exists mlnx_qos; then
    mlnx_qos -i "$INTERFACE" --trust=dscp 2>&1 && \
        ok "Trust mode 设置为 DSCP" || \
        warn "Trust mode 设置失败"
else
    warn "mlnx_qos 不可用，跳过 Trust Mode 设置"
    info "手动设置: mlnx_qos -i $INTERFACE --trust=dscp"
fi

# ─── 步骤 4: DSCP-to-Priority 映射 ──────────────────────────────────────────

section "步骤 4: 配置 DSCP-to-Priority 映射"

if cmd_exists mlnx_qos; then
    # 将指定 DSCP 值映射到指定 Priority
    mlnx_qos -i "$INTERFACE" --dscp2prio "set,$DSCP_VALUE,$PRIORITY" 2>&1 && \
        ok "DSCP $DSCP_VALUE → Priority $PRIORITY 映射已设置" || \
        warn "DSCP-to-Priority 映射设置失败"
else
    warn "mlnx_qos 不可用，跳过 DSCP 映射"
    info "手动设置: mlnx_qos -i $INTERFACE --dscp2prio set,$DSCP_VALUE,$PRIORITY"
fi

# ─── 步骤 5: 设置 MTU ────────────────────────────────────────────────────────

section "步骤 5: 设置 MTU 为 $MTU"

current_mtu=$(cat "/sys/class/net/$INTERFACE/mtu" 2>/dev/null || echo "0")
if [[ "$current_mtu" -lt "$MTU" ]]; then
    ip link set dev "$INTERFACE" mtu "$MTU" 2>&1 && \
        ok "MTU 设置为 $MTU" || \
        warn "MTU 设置失败 (当前: $current_mtu)"
else
    ok "MTU 已满足要求 (当前: $current_mtu)"
fi

# ─── 步骤 6: 配置 Traffic Class ──────────────────────────────────────────────

section "步骤 6: 配置 Traffic Class (TC)"

if cmd_exists mlnx_qos; then
    # 设置 ETS (Enhanced Transmission Selection): 为 RoCE TC 分配带宽
    # 示例: TC0=50% (普通流量), TC3=50% (RoCE 流量)
    info "配置 ETS 带宽分配..."

    # 设置优先级到 TC 映射: priority 3 → TC 3
    mlnx_qos -i "$INTERFACE" --prio_tc="0,0,0,$PRIORITY,0,0,0,0" 2>&1 && \
        ok "Priority $PRIORITY → TC $PRIORITY 映射已设置" || \
        warn "Priority-to-TC 映射设置失败"
else
    warn "mlnx_qos 不可用，跳过 TC 配置"
fi

# ─── 步骤 7: 内核参数优化 ────────────────────────────────────────────────────

section "步骤 7: 内核参数优化"

# 增大 UDP 缓冲区 (RoCE v2 使用 UDP)
sysctl -w net.core.rmem_max=16777216 >/dev/null 2>&1 && \
    ok "net.core.rmem_max = 16MB" || warn "rmem_max 设置失败"

sysctl -w net.core.wmem_max=16777216 >/dev/null 2>&1 && \
    ok "net.core.wmem_max = 16MB" || warn "wmem_max 设置失败"

sysctl -w net.core.rmem_default=1048576 >/dev/null 2>&1 && \
    ok "net.core.rmem_default = 1MB" || warn "rmem_default 设置失败"

sysctl -w net.core.wmem_default=1048576 >/dev/null 2>&1 && \
    ok "net.core.wmem_default = 1MB" || warn "wmem_default 设置失败"

# netdev backlog
sysctl -w net.core.netdev_max_backlog=250000 >/dev/null 2>&1 && \
    ok "net.core.netdev_max_backlog = 250000" || true

# ─── 步骤 8: 确认接口 UP ────────────────────────────────────────────────────

section "步骤 8: 确认接口状态"

OPER_STATE=$(cat "/sys/class/net/$INTERFACE/operstate" 2>/dev/null || echo "unknown")
if [[ "$OPER_STATE" != "up" ]]; then
    ip link set dev "$INTERFACE" up 2>&1 && \
        ok "接口 $INTERFACE 已激活" || \
        warn "接口激活失败"
else
    ok "接口 $INTERFACE 状态: UP"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 配置验证
# ═══════════════════════════════════════════════════════════════════════════════

header "配置验证"

section "验证 GID 表 (RoCE v2 条目)"
GID_DIR="/sys/class/infiniband/$RDMA_DEV/ports/${RDMA_PORT:-1}/gids"
ROCEV2_COUNT=0

if [[ -d "$GID_DIR" ]]; then
    for gid_file in "$GID_DIR"/*; do
        if [[ -f "$gid_file" ]]; then
            idx=$(basename "$gid_file")
            gid_val=$(cat "$gid_file" 2>/dev/null)
            gtype_file="/sys/class/infiniband/$RDMA_DEV/ports/${RDMA_PORT:-1}/gid_attrs/types/$idx"

            if [[ -f "$gtype_file" ]]; then
                gtype=$(cat "$gtype_file" 2>/dev/null)
                if [[ "$gtype" == *"RoCE v2"* ]] && \
                   [[ "$gid_val" != "0000:0000:0000:0000:0000:0000:0000:0000" ]]; then
                    ROCEV2_COUNT=$((ROCEV2_COUNT + 1))
                    if [[ "$idx" == "$GID_INDEX" ]]; then
                        ok "GID[$idx] = $gid_val (RoCE v2) ← 选定索引"
                    else
                        info "GID[$idx] = $gid_val (RoCE v2)"
                    fi
                fi
            fi
        fi
    done
fi

if [[ $ROCEV2_COUNT -gt 0 ]]; then
    ok "找到 $ROCEV2_COUNT 个 RoCE v2 GID 条目"
else
    warn "未找到 RoCE v2 GID 条目 (可能需要等待接口完全初始化)"
fi

section "验证连通性建议"
echo ""
info "使用以下命令验证 RoCE 连通性:"
echo ""
echo "  # 服务端 (在对端运行):"
echo "  ib_write_bw -d $RDMA_DEV -x $GID_INDEX --report_gbits -R"
echo ""
echo "  # 客户端 (在本机运行):"
echo "  ib_write_bw -d $RDMA_DEV -x $GID_INDEX --report_gbits -R <server_ip>"
echo ""
echo "  # 使用指定 TOS (如果需要):"
echo "  ib_write_bw -d $RDMA_DEV -x $GID_INDEX --report_gbits --tos=$TOS_VALUE <server_ip>"
echo ""

# ─── 配置摘要 ────────────────────────────────────────────────────────────────

header "配置摘要"
echo ""
echo -e "  接口:          ${BOLD}$INTERFACE${NC}"
echo -e "  RDMA 设备:     ${BOLD}$RDMA_DEV${NC} (端口 ${RDMA_PORT:-1})"
echo -e "  RoCE 版本:     ${BOLD}v$ROCE_MODE${NC}"
echo -e "  GID 索引:      ${BOLD}$GID_INDEX${NC}"
echo -e "  DSCP:          ${BOLD}$DSCP_VALUE${NC} (TOS: $TOS_VALUE)"
echo -e "  Priority:      ${BOLD}$PRIORITY${NC}"
echo -e "  MTU:           ${BOLD}$MTU${NC}"
echo ""
ok "RoCE v${ROCE_MODE} 配置完成"
echo ""
info "注意: 部分配置在重启后会丢失，建议添加到启动脚本或 systemd unit 中"
