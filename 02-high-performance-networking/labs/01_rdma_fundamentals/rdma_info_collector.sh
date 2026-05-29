#!/usr/bin/env bash
###############################################################################
# rdma_info_collector.sh - RDMA 设备信息收集脚本
#
# 功能: 收集系统中所有 RDMA 设备的详细信息，包括：
#   - 设备列表与基本信息 (ibv_devices / ibv_devinfo)
#   - InfiniBand 端口状态 (ibstat / ibstatus)
#   - RDMA 链路信息 (rdma link / rdma dev)
#   - 驱动与固件版本
#   - 网络接口映射
#   - 性能计数器
#
# 用法: sudo ./rdma_info_collector.sh [--output <file>] [--json]
###############################################################################

set -euo pipefail

# ─── 颜色定义 ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ─── 默认参数 ────────────────────────────────────────────────────────────────
OUTPUT_FILE=""
JSON_MODE=false
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
HOSTNAME_STR=$(hostname)

# ─── 参数解析 ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --json|-j)
            JSON_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--output <file>] [--json]"
            echo ""
            echo "Options:"
            echo "  --output, -o <file>   将结果保存到指定文件"
            echo "  --json, -j            输出 JSON 格式"
            echo "  --help, -h            显示帮助信息"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ─── 工具函数 ────────────────────────────────────────────────────────────────

print_header() {
    local title="$1"
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $title${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
}

print_section() {
    local title="$1"
    echo -e "\n${GREEN}── $title ──${NC}"
}

print_warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_ok() {
    echo -e "${GREEN}[OK] $1${NC}"
}

cmd_exists() {
    command -v "$1" &>/dev/null
}

run_cmd() {
    local cmd="$1"
    local description="${2:-$cmd}"
    echo -e "${YELLOW}$ $cmd${NC}"
    if eval "$cmd" 2>&1; then
        return 0
    else
        print_warn "$description 命令执行失败或无输出"
        return 1
    fi
}

# ─── 开始收集 ────────────────────────────────────────────────────────────────

collect_info() {

print_header "RDMA 设备信息收集报告"
echo "主机名: $HOSTNAME_STR"
echo "时间戳: $TIMESTAMP"
echo "内核版本: $(uname -r)"

# ═══════════════════════════════════════════════════════════════
# 1. 前置检查
# ═══════════════════════════════════════════════════════════════
print_header "1. 前置环境检查"

print_section "1.1 RDMA 相关内核模块"
run_cmd "lsmod | grep -E 'rdma|ib_|mlx|rxe|roce' | sort" "内核模块检查" || \
    print_warn "未检测到 RDMA 相关内核模块"

print_section "1.2 RDMA 用户空间工具检查"
TOOLS=("ibv_devices" "ibv_devinfo" "ibstat" "ibstatus" "rdma" "ibdiagnet" "perftest" "ib_write_bw")
for tool in "${TOOLS[@]}"; do
    if cmd_exists "$tool"; then
        print_ok "$tool 可用 ($(which $tool))"
    else
        print_warn "$tool 未安装"
    fi
done

print_section "1.3 rdma-core 包信息"
if cmd_exists rpm; then
    run_cmd "rpm -qa | grep -i rdma" "RPM rdma 包" || true
elif cmd_exists dpkg; then
    run_cmd "dpkg -l | grep -i rdma" "DEB rdma 包" || true
fi

# ═══════════════════════════════════════════════════════════════
# 2. ibv_devices — RDMA 设备列表
# ═══════════════════════════════════════════════════════════════
print_header "2. RDMA 设备列表 (ibv_devices)"

if cmd_exists ibv_devices; then
    run_cmd "ibv_devices" "ibv_devices"
else
    print_error "ibv_devices 不可用，请安装 libibverbs-utils / rdma-core"
fi

# ═══════════════════════════════════════════════════════════════
# 3. ibv_devinfo — 设备详细信息
# ═══════════════════════════════════════════════════════════════
print_header "3. RDMA 设备详细信息 (ibv_devinfo)"

if cmd_exists ibv_devinfo; then
    run_cmd "ibv_devinfo -v" "ibv_devinfo 详细信息"
else
    print_error "ibv_devinfo 不可用"
fi

# ═══════════════════════════════════════════════════════════════
# 4. ibstat — InfiniBand 端口状态
# ═══════════════════════════════════════════════════════════════
print_header "4. InfiniBand 端口状态 (ibstat)"

if cmd_exists ibstat; then
    run_cmd "ibstat" "ibstat"
else
    print_warn "ibstat 不可用 (infiniband-diags 未安装)"
fi

# ═══════════════════════════════════════════════════════════════
# 5. ibstatus — 简洁端口状态
# ═══════════════════════════════════════════════════════════════
print_header "5. 端口状态摘要 (ibstatus)"

if cmd_exists ibstatus; then
    run_cmd "ibstatus" "ibstatus"
else
    print_warn "ibstatus 不可用"
fi

# ═══════════════════════════════════════════════════════════════
# 6. rdma 工具 — 现代 iproute2 RDMA 管理
# ═══════════════════════════════════════════════════════════════
print_header "6. RDMA 链路与设备 (rdma 工具)"

if cmd_exists rdma; then
    print_section "6.1 rdma dev show"
    run_cmd "rdma dev show" "rdma dev"

    print_section "6.2 rdma link show"
    run_cmd "rdma link show" "rdma link"

    print_section "6.3 rdma resource show"
    run_cmd "rdma resource show" "rdma resource"

    print_section "6.4 rdma statistic show"
    run_cmd "rdma statistic show" "rdma statistic" || true

    print_section "6.5 rdma system show"
    run_cmd "rdma system show" "rdma system" || true
else
    print_warn "rdma 工具不可用 (iproute2 版本可能过旧)"
fi

# ═══════════════════════════════════════════════════════════════
# 7. 网络接口映射
# ═══════════════════════════════════════════════════════════════
print_header "7. RDMA 设备与网络接口映射"

print_section "7.1 /sys/class/infiniband/ 设备列表"
if [[ -d /sys/class/infiniband ]]; then
    for dev in /sys/class/infiniband/*; do
        dev_name=$(basename "$dev")
        echo -e "${CYAN}设备: $dev_name${NC}"

        # 固件版本
        if [[ -f "$dev/fw_ver" ]]; then
            echo "  固件版本: $(cat "$dev/fw_ver")"
        fi

        # 板卡 ID
        if [[ -f "$dev/board_id" ]]; then
            echo "  板卡 ID: $(cat "$dev/board_id")"
        fi

        # HCA 类型
        if [[ -f "$dev/hca_type" ]]; then
            echo "  HCA 类型: $(cat "$dev/hca_type")"
        fi

        # Node GUID
        if [[ -f "$dev/node_guid" ]]; then
            echo "  Node GUID: $(cat "$dev/node_guid")"
        fi

        # 端口信息
        for port_dir in "$dev/ports"/*/; do
            if [[ -d "$port_dir" ]]; then
                port_num=$(basename "$port_dir")
                echo "  端口 $port_num:"

                if [[ -f "$port_dir/state" ]]; then
                    echo "    状态: $(cat "$port_dir/state")"
                fi
                if [[ -f "$port_dir/phys_state" ]]; then
                    echo "    物理状态: $(cat "$port_dir/phys_state")"
                fi
                if [[ -f "$port_dir/rate" ]]; then
                    echo "    速率: $(cat "$port_dir/rate")"
                fi
                if [[ -f "$port_dir/link_layer" ]]; then
                    echo "    链路层: $(cat "$port_dir/link_layer")"
                fi
                if [[ -f "$port_dir/lid" ]]; then
                    echo "    LID: $(cat "$port_dir/lid")"
                fi
                if [[ -f "$port_dir/sm_lid" ]]; then
                    echo "    SM LID: $(cat "$port_dir/sm_lid")"
                fi

                # GID 表 (前几条)
                if [[ -d "$port_dir/gids" ]]; then
                    echo "    GID 表 (前 5 条):"
                    local gid_count=0
                    for gid_file in "$port_dir/gids"/*; do
                        if [[ -f "$gid_file" ]]; then
                            gid_val=$(cat "$gid_file")
                            if [[ "$gid_val" != "0000:0000:0000:0000:0000:0000:0000:0000" ]]; then
                                gid_idx=$(basename "$gid_file")
                                echo "      [$gid_idx] $gid_val"
                                gid_count=$((gid_count + 1))
                                [[ $gid_count -ge 5 ]] && break
                            fi
                        fi
                    done
                fi

                # 关联网络接口
                if [[ -d "$port_dir/gid_attrs/ndevs" ]]; then
                    for ndev_file in "$port_dir/gid_attrs/ndevs"/*; do
                        if [[ -f "$ndev_file" ]]; then
                            ndev=$(cat "$ndev_file" 2>/dev/null)
                            if [[ -n "$ndev" ]]; then
                                echo "    关联网卡: $ndev"
                                break
                            fi
                        fi
                    done
                fi
            fi
        done
        echo ""
    done
else
    print_warn "/sys/class/infiniband 目录不存在"
fi

print_section "7.2 RDMA netdev 关联"
if cmd_exists rdma; then
    run_cmd "rdma link show -jp 2>/dev/null || rdma link show" "rdma link (详细)" || true
fi

# ═══════════════════════════════════════════════════════════════
# 8. 驱动与固件信息
# ═══════════════════════════════════════════════════════════════
print_header "8. 驱动与固件信息"

print_section "8.1 Mellanox OFED 版本"
if [[ -f /sys/module/mlx5_core/version ]]; then
    echo "mlx5_core 版本: $(cat /sys/module/mlx5_core/version)"
fi

if cmd_exists ofed_info; then
    run_cmd "ofed_info -s" "OFED 版本"
else
    print_warn "ofed_info 不可用 (MLNX_OFED 可能未安装)"
fi

print_section "8.2 ethtool 驱动信息"
if cmd_exists ethtool; then
    for iface in $(ls /sys/class/net/ 2>/dev/null); do
        if [[ -d "/sys/class/net/$iface/device/infiniband" ]] 2>/dev/null; then
            echo -e "${CYAN}接口: $iface${NC}"
            run_cmd "ethtool -i $iface" "ethtool 驱动信息" || true
        fi
    done
fi

# ═══════════════════════════════════════════════════════════════
# 9. 性能相关配置
# ═══════════════════════════════════════════════════════════════
print_header "9. 性能相关配置"

print_section "9.1 Hugepages 配置"
echo "HugePages_Total: $(cat /proc/sys/vm/nr_hugepages 2>/dev/null || echo N/A)"
echo "HugePages_Free:  $(grep HugePages_Free /proc/meminfo 2>/dev/null | awk '{print $2}' || echo N/A)"
echo "Hugepagesize:    $(grep Hugepagesize /proc/meminfo 2>/dev/null | awk '{print $2, $3}' || echo N/A)"

print_section "9.2 NUMA 拓扑"
if cmd_exists numactl; then
    run_cmd "numactl -H" "NUMA 硬件信息" || true
fi

# 显示 RDMA 设备的 NUMA 节点
for dev in /sys/class/infiniband/*; do
    if [[ -d "$dev" ]]; then
        dev_name=$(basename "$dev")
        numa_node=$(cat "$dev/device/numa_node" 2>/dev/null || echo "N/A")
        echo "设备 $dev_name NUMA 节点: $numa_node"
    fi
done

print_section "9.3 CPU Governor"
if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    echo "CPU Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
else
    echo "CPU Governor: 信息不可用"
fi

print_section "9.4 IRQ 亲和性 (RDMA 设备)"
for dev in /sys/class/infiniband/*; do
    dev_name=$(basename "$dev")
    pci_dev=$(readlink -f "$dev/device" 2>/dev/null | xargs basename 2>/dev/null)
    if [[ -n "$pci_dev" ]]; then
        irq_count=$(ls /proc/irq/*/smp_affinity 2>/dev/null | \
                    xargs grep -l "" 2>/dev/null | \
                    xargs -I {} dirname {} | \
                    xargs -I {} cat {}/../../$pci_dev 2>/dev/null | wc -l || echo 0)
        echo "设备 $dev_name (PCI: $pci_dev): 关联 IRQ 检查完成"
    fi
done

# ═══════════════════════════════════════════════════════════════
# 10. 摘要
# ═══════════════════════════════════════════════════════════════
print_header "10. 收集摘要"

device_count=0
if [[ -d /sys/class/infiniband ]]; then
    device_count=$(ls /sys/class/infiniband/ 2>/dev/null | wc -w)
fi

echo "检测到 RDMA 设备数量: $device_count"
echo "报告生成时间: $(date)"
echo "报告主机: $HOSTNAME_STR"
echo ""

if [[ $device_count -eq 0 ]]; then
    print_warn "未检测到任何 RDMA 设备。可能原因："
    echo "  1. 未安装 RDMA 网卡"
    echo "  2. 驱动未加载 (尝试: modprobe mlx5_ib)"
    echo "  3. 使用软件模拟 (尝试: modprobe rdma_rxe)"
    echo "  4. 运行在虚拟机/容器中且未透传设备"
else
    print_ok "信息收集完成"
fi

}

# ─── 主入口 ───────────────────────────────────────────────────────────────────

if [[ -n "$OUTPUT_FILE" ]]; then
    collect_info 2>&1 | tee "$OUTPUT_FILE"
    echo ""
    print_ok "报告已保存到: $OUTPUT_FILE"
else
    collect_info 2>&1
fi
