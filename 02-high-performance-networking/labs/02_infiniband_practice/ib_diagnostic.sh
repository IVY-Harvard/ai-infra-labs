#!/usr/bin/env bash
###############################################################################
# ib_diagnostic.sh - InfiniBand 网络综合诊断脚本
#
# 功能: 对 InfiniBand 网络进行全面诊断，包括：
#   - ibstat / ibstatus 端口状态检查
#   - ibdiagnet 网络拓扑与健康诊断
#   - perftest 带宽与延迟快速测试
#   - 错误计数器检查
#   - SM 状态验证
#
# 用法: sudo ./ib_diagnostic.sh [--full] [--perf <server_ip>] [--output <dir>]
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

# ─── 参数 ────────────────────────────────────────────────────────────────────
FULL_DIAG=false
PERF_SERVER=""
OUTPUT_DIR=""
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

while [[ $# -gt 0 ]]; do
    case $1 in
        --full|-f)
            FULL_DIAG=true
            shift
            ;;
        --perf|-p)
            PERF_SERVER="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--full] [--perf <server_ip>] [--output <dir>]"
            echo ""
            echo "Options:"
            echo "  --full, -f              运行完整诊断 (包括 ibdiagnet)"
            echo "  --perf, -p <server_ip>  对指定服务器运行性能测试"
            echo "  --output, -o <dir>      保存诊断结果到指定目录"
            echo "  --help, -h              显示帮助"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1
            ;;
    esac
done

if [[ -n "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
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

save_output() {
    local name="$1"
    if [[ -n "$OUTPUT_DIR" ]]; then
        cat > "$OUTPUT_DIR/${name}_${TIMESTAMP}.txt"
    else
        cat
    fi
}

# ─── 诊断计数器 ──────────────────────────────────────────────────────────────
TOTAL_CHECKS=0
PASS_CHECKS=0
WARN_CHECKS=0
FAIL_CHECKS=0

check_pass() { TOTAL_CHECKS=$((TOTAL_CHECKS+1)); PASS_CHECKS=$((PASS_CHECKS+1)); ok "$1"; }
check_warn() { TOTAL_CHECKS=$((TOTAL_CHECKS+1)); WARN_CHECKS=$((WARN_CHECKS+1)); warn "$1"; }
check_fail() { TOTAL_CHECKS=$((TOTAL_CHECKS+1)); FAIL_CHECKS=$((FAIL_CHECKS+1)); err "$1"; }

# ═══════════════════════════════════════════════════════════════════════════════
# 开始诊断
# ═══════════════════════════════════════════════════════════════════════════════

header "InfiniBand 网络诊断 — $(hostname) — $(date)"

# ───────────────────────────────────────────────────────────────
# 1. 基础环境检查
# ───────────────────────────────────────────────────────────────
header "1. 基础环境检查"

section "1.1 必要工具检测"
REQUIRED_TOOLS=("ibstat" "ibstatus" "ibv_devinfo" "perfquery")
OPTIONAL_TOOLS=("ibdiagnet" "ibnetdiscover" "sminfo" "smpquery" "ibswitches" "iblinkinfo")

for tool in "${REQUIRED_TOOLS[@]}"; do
    if cmd_exists "$tool"; then
        check_pass "$tool 可用"
    else
        check_fail "$tool 不可用 — 请安装 infiniband-diags"
    fi
done

for tool in "${OPTIONAL_TOOLS[@]}"; do
    if cmd_exists "$tool"; then
        check_pass "$tool 可用 (可选)"
    else
        check_warn "$tool 不可用 (可选工具)"
    fi
done

section "1.2 内核模块"
for mod in ib_core ib_uverbs mlx5_ib mlx5_core; do
    if lsmod | grep -qw "$mod"; then
        check_pass "模块 $mod 已加载"
    else
        check_warn "模块 $mod 未加载"
    fi
done

# ───────────────────────────────────────────────────────────────
# 2. ibstat — 端口状态
# ───────────────────────────────────────────────────────────────
header "2. IB 端口状态 (ibstat)"

if cmd_exists ibstat; then
    IBSTAT_OUTPUT=$(ibstat 2>&1)
    echo "$IBSTAT_OUTPUT" | save_output "ibstat"
    echo "$IBSTAT_OUTPUT"
    echo ""

    # 解析端口状态
    while IFS= read -r line; do
        if [[ "$line" =~ State:\ +(.*) ]]; then
            state="${BASH_REMATCH[1]}"
            if [[ "$state" == "Active" ]]; then
                check_pass "端口状态: Active"
            elif [[ "$state" == "Initializing" ]]; then
                check_warn "端口状态: Initializing (等待 SM)"
            else
                check_fail "端口状态: $state"
            fi
        fi
        if [[ "$line" =~ Physical\ state:\ +(.*) ]]; then
            phys="${BASH_REMATCH[1]}"
            if [[ "$phys" == "LinkUp" ]]; then
                check_pass "物理状态: LinkUp"
            else
                check_fail "物理状态: $phys"
            fi
        fi
        if [[ "$line" =~ Rate:\ +(.*) ]]; then
            info "链路速率: ${BASH_REMATCH[1]}"
        fi
    done <<< "$IBSTAT_OUTPUT"
else
    check_fail "ibstat 不可用"
fi

# ───────────────────────────────────────────────────────────────
# 3. ibstatus — 补充状态信息
# ───────────────────────────────────────────────────────────────
header "3. IB 端口摘要 (ibstatus)"

if cmd_exists ibstatus; then
    ibstatus 2>&1 | save_output "ibstatus"
    ibstatus 2>&1
fi

# ───────────────────────────────────────────────────────────────
# 4. Subnet Manager 状态
# ───────────────────────────────────────────────────────────────
header "4. Subnet Manager 状态"

section "4.1 SM 信息"
if cmd_exists sminfo; then
    SM_OUT=$(sminfo 2>&1) || true
    if [[ -n "$SM_OUT" && ! "$SM_OUT" =~ "error" ]]; then
        echo "$SM_OUT"
        check_pass "SM 可达"
    else
        check_fail "SM 不可达或未运行"
        info "尝试: systemctl start opensm"
    fi
else
    check_warn "sminfo 不可用"
fi

section "4.2 SM 进程检查"
if pgrep -x opensm &>/dev/null; then
    check_pass "OpenSM 进程运行中 (PID: $(pgrep -x opensm))"
else
    check_warn "OpenSM 进程未在本机运行 (可能运行在交换机上)"
fi

section "4.3 本地 SM 配置"
OPENSM_CONF="/etc/opensm/opensm.conf"
if [[ -f "$OPENSM_CONF" ]]; then
    check_pass "OpenSM 配置文件存在: $OPENSM_CONF"
    info "路由引擎: $(grep -E '^routing_engine' $OPENSM_CONF 2>/dev/null | head -1 || echo '默认')"
    info "SM 优先级: $(grep -E '^sm_priority' $OPENSM_CONF 2>/dev/null | head -1 || echo '默认')"
else
    info "OpenSM 配置文件不存在 (SM 可能运行在交换机上)"
fi

# ───────────────────────────────────────────────────────────────
# 5. 错误计数器检查
# ───────────────────────────────────────────────────────────────
header "5. 端口错误计数器"

if cmd_exists perfquery; then
    section "5.1 本地端口计数器"

    # 获取所有 IB 设备和端口
    for dev_dir in /sys/class/infiniband/*/; do
        dev_name=$(basename "$dev_dir")
        for port_dir in "$dev_dir/ports"/*/; do
            if [[ -d "$port_dir" ]]; then
                port_num=$(basename "$port_dir")
                link_layer=$(cat "$port_dir/link_layer" 2>/dev/null || echo "unknown")

                # 只对 InfiniBand 链路执行 perfquery
                if [[ "$link_layer" == "InfiniBand" ]]; then
                    echo -e "\n${CYAN}设备 $dev_name 端口 $port_num:${NC}"
                    PERF_OUT=$(perfquery -d "$dev_name" -P "$port_num" 2>&1) || true

                    if [[ -n "$PERF_OUT" ]]; then
                        echo "$PERF_OUT"

                        # 检查关键错误计数器
                        sym_err=$(echo "$PERF_OUT" | grep "SymbolErrorCounter" | awk -F'.' '{print $NF}' | tr -d ' ')
                        link_err=$(echo "$PERF_OUT" | grep "LinkErrorRecoveryCounter" | awk -F'.' '{print $NF}' | tr -d ' ')
                        rcv_err=$(echo "$PERF_OUT" | grep "PortRcvErrors" | awk -F'.' '{print $NF}' | tr -d ' ')
                        xmit_disc=$(echo "$PERF_OUT" | grep "PortXmitDiscards" | awk -F'.' '{print $NF}' | tr -d ' ')

                        [[ "${sym_err:-0}" -gt 0 ]] 2>/dev/null && check_warn "SymbolErrors: $sym_err" || true
                        [[ "${link_err:-0}" -gt 0 ]] 2>/dev/null && check_warn "LinkErrorRecovery: $link_err" || true
                        [[ "${rcv_err:-0}" -gt 0 ]] 2>/dev/null && check_fail "PortRcvErrors: $rcv_err" || true
                        [[ "${xmit_disc:-0}" -gt 0 ]] 2>/dev/null && check_warn "PortXmitDiscards: $xmit_disc" || true
                    fi
                fi
            fi
        done
    done

    section "5.2 扩展计数器"
    for dev_dir in /sys/class/infiniband/*/; do
        dev_name=$(basename "$dev_dir")
        for port_dir in "$dev_dir/ports"/*/; do
            if [[ -d "$port_dir" ]]; then
                port_num=$(basename "$port_dir")
                link_layer=$(cat "$port_dir/link_layer" 2>/dev/null || echo "unknown")
                if [[ "$link_layer" == "InfiniBand" ]]; then
                    echo -e "\n${CYAN}设备 $dev_name 端口 $port_num (扩展):${NC}"
                    perfquery -d "$dev_name" -P "$port_num" -x 2>&1 || true
                fi
            fi
        done
    done
else
    check_fail "perfquery 不可用"
fi

# ───────────────────────────────────────────────────────────────
# 6. 网络拓扑
# ───────────────────────────────────────────────────────────────
header "6. 网络拓扑"

section "6.1 IB 交换机列表"
if cmd_exists ibswitches; then
    ibswitches 2>&1 || check_warn "无法获取交换机列表"
fi

section "6.2 网络发现"
if cmd_exists ibnetdiscover; then
    info "运行 ibnetdiscover (可能需要几秒)..."
    ibnetdiscover 2>&1 | head -50 || check_warn "ibnetdiscover 失败"
    info "(仅显示前 50 行)"
fi

section "6.3 链路信息"
if cmd_exists iblinkinfo; then
    iblinkinfo 2>&1 | head -30 || check_warn "iblinkinfo 失败"
fi

# ───────────────────────────────────────────────────────────────
# 7. ibdiagnet 完整诊断 (--full 模式)
# ───────────────────────────────────────────────────────────────
if [[ "$FULL_DIAG" == true ]]; then
    header "7. ibdiagnet 完整网络诊断"

    if cmd_exists ibdiagnet; then
        DIAG_DIR="${OUTPUT_DIR:-/tmp}/ibdiagnet_${TIMESTAMP}"
        mkdir -p "$DIAG_DIR"

        info "运行 ibdiagnet (可能需要数分钟)..."
        info "输出目录: $DIAG_DIR"

        ibdiagnet --output "$DIAG_DIR" 2>&1 | tail -20

        if [[ -f "$DIAG_DIR/ibdiagnet2.log" ]]; then
            section "诊断摘要"
            grep -E "(ERROR|WARNING|INFO)" "$DIAG_DIR/ibdiagnet2.log" | tail -20 || true
            check_pass "ibdiagnet 完成，完整报告: $DIAG_DIR"
        fi
    else
        check_warn "ibdiagnet 不可用 (需要安装 ibutils2)"
    fi
fi

# ───────────────────────────────────────────────────────────────
# 8. 性能测试 (--perf 模式)
# ───────────────────────────────────────────────────────────────
if [[ -n "$PERF_SERVER" ]]; then
    header "8. 性能测试 (目标: $PERF_SERVER)"

    section "8.1 RDMA Write 带宽"
    if cmd_exists ib_write_bw; then
        info "测试 RDMA Write 带宽..."
        ib_write_bw -d mlx5_0 --report_gbits -D 5 "$PERF_SERVER" 2>&1 || \
            check_warn "ib_write_bw 测试失败 (请确保服务端已运行: ib_write_bw -d mlx5_0)"
    fi

    section "8.2 RDMA Write 延迟"
    if cmd_exists ib_write_lat; then
        info "测试 RDMA Write 延迟..."
        ib_write_lat -d mlx5_0 -n 1000 "$PERF_SERVER" 2>&1 || \
            check_warn "ib_write_lat 测试失败"
    fi

    section "8.3 RDMA Send 带宽"
    if cmd_exists ib_send_bw; then
        info "测试 RDMA Send 带宽..."
        ib_send_bw -d mlx5_0 --report_gbits -D 5 "$PERF_SERVER" 2>&1 || \
            check_warn "ib_send_bw 测试失败"
    fi

    section "8.4 RDMA Read 带宽"
    if cmd_exists ib_read_bw; then
        info "测试 RDMA Read 带宽..."
        ib_read_bw -d mlx5_0 --report_gbits -D 5 "$PERF_SERVER" 2>&1 || \
            check_warn "ib_read_bw 测试失败"
    fi

    section "8.5 Atomic 延迟"
    if cmd_exists ib_atomic_lat; then
        info "测试 Atomic 延迟..."
        ib_atomic_lat -d mlx5_0 -n 1000 "$PERF_SERVER" 2>&1 || \
            check_warn "ib_atomic_lat 测试失败"
    fi
fi

# ───────────────────────────────────────────────────────────────
# 诊断总结
# ───────────────────────────────────────────────────────────────
header "诊断总结"
echo ""
echo -e "  总检查项: ${BOLD}$TOTAL_CHECKS${NC}"
echo -e "  ${GREEN}通过: $PASS_CHECKS${NC}"
echo -e "  ${YELLOW}警告: $WARN_CHECKS${NC}"
echo -e "  ${RED}失败: $FAIL_CHECKS${NC}"
echo ""

if [[ $FAIL_CHECKS -gt 0 ]]; then
    err "存在 $FAIL_CHECKS 个失败项，请检查上述输出"
    exit 1
elif [[ $WARN_CHECKS -gt 0 ]]; then
    warn "存在 $WARN_CHECKS 个警告项"
    exit 0
else
    ok "所有检查通过"
    exit 0
fi
