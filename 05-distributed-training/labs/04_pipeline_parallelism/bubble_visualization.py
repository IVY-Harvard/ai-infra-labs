"""
Lab 04 - Pipeline Bubble 可视化
================================
在终端中可视化 GPipe 和 1F1B 的调度时间线。

运行:
    python bubble_visualization.py

（注意：这个脚本不需要多 GPU，只是可视化调度逻辑）
"""

import sys


def visualize_gpipe(p: int, m: int):
    """
    可视化 GPipe 调度。
    p: pipeline stages
    m: micro-batches
    """
    # 计算每个 stage 在每个时间步做什么
    total_steps = 2 * (m + p - 1)
    schedule = [["  "] * total_steps for _ in range(p)]

    # Forward phase
    for mb in range(m):
        for stage in range(p):
            t = mb + stage
            schedule[stage][t] = f"F{mb}"

    # Backward phase (紧接 forward 之后)
    bwd_start = m + p - 1
    for mb in reversed(range(m)):
        for stage in reversed(range(p)):
            t = bwd_start + (m - 1 - mb) + (p - 1 - stage)
            if t < total_steps:
                schedule[stage][t] = f"B{mb}"

    # 打印
    print(f"\n{'='*70}")
    print(f"GPipe Schedule (stages={p}, micro-batches={m})")
    print(f"Bubble 率: {(p-1)/(m+p-1):.1%}")
    print(f"{'='*70}")

    # 时间轴
    header = "       "
    for t in range(min(total_steps, 40)):
        header += f"{t:>4}"
    print(header)
    print("       " + "----" * min(total_steps, 40))

    for stage in range(p):
        line = f"GPU {stage}: "
        for t in range(min(total_steps, 40)):
            cell = schedule[stage][t]
            if cell.startswith("F"):
                line += f"\033[92m{cell:>4}\033[0m"  # 绿色
            elif cell.startswith("B"):
                line += f"\033[94m{cell:>4}\033[0m"  # 蓝色
            else:
                line += f"\033[90m{'##':>4}\033[0m"  # 灰色 (bubble)
        print(line)

    # 统计
    total_cells = p * total_steps
    active_cells = sum(1 for s in schedule for c in s if c.strip())
    bubble_cells = total_cells - active_cells
    print(f"\n  总时间步: {total_steps}")
    print(f"  活跃: {active_cells}, Bubble: {bubble_cells}")
    print(f"  实际 Bubble 率: {bubble_cells/total_cells:.1%}")


def visualize_1f1b(p: int, m: int):
    """
    可视化 1F1B 调度。
    """
    total_steps = 2 * m + 2 * (p - 1)
    schedule = [["  "] * total_steps for _ in range(p)]

    for stage in range(p):
        fwd_idx = 0
        bwd_idx = 0

        # warmup: stage 越靠前，warmup 越多
        num_warmup = p - stage - 1
        t = stage  # 开始时间 = stage 编号

        # Warmup forwards
        for _ in range(min(num_warmup, m)):
            if t < total_steps:
                schedule[stage][t] = f"F{fwd_idx}"
            fwd_idx += 1
            t += 1

        # Steady: 1F1B
        while fwd_idx < m:
            if t < total_steps:
                schedule[stage][t] = f"F{fwd_idx}"
            fwd_idx += 1
            t += 1
            if t < total_steps and bwd_idx < m:
                schedule[stage][t] = f"B{bwd_idx}"
            bwd_idx += 1
            t += 1

        # Cooldown: remaining backwards
        while bwd_idx < m:
            if t < total_steps:
                schedule[stage][t] = f"B{bwd_idx}"
            bwd_idx += 1
            t += 1

    # 打印
    print(f"\n{'='*70}")
    print(f"1F1B Schedule (stages={p}, micro-batches={m})")
    print(f"Bubble 率: {(p-1)/(m+p-1):.1%}")
    print(f"最大同时激活数: {p} (vs GPipe 的 {m})")
    print(f"{'='*70}")

    max_display = min(max(2 * m + 2 * p, 30), 50)

    header = "       "
    for t in range(max_display):
        header += f"{t:>4}"
    print(header)
    print("       " + "----" * max_display)

    for stage in range(p):
        line = f"GPU {stage}: "
        for t in range(max_display):
            cell = schedule[stage][t] if t < total_steps else "  "
            if cell.startswith("F"):
                line += f"\033[92m{cell:>4}\033[0m"
            elif cell.startswith("B"):
                line += f"\033[94m{cell:>4}\033[0m"
            else:
                line += f"\033[90m{'##':>4}\033[0m"
        print(line)


def compare_memory(p: int, m: int):
    """对比 GPipe 和 1F1B 的激活值显存"""
    print(f"\n{'='*70}")
    print(f"显存对比 (stages={p}, micro-batches={m})")
    print(f"{'='*70}")

    # 假设每个 micro-batch 的激活值 = 1 个单位
    print(f"  GPipe 激活值峰值: {m} 个 micro-batch (所有前向完成时)")
    print(f"  1F1B  激活值峰值: {p} 个 micro-batch (warmup 完成时)")
    print(f"  节省比例: {(1 - p/m)*100:.0f}% (当 m >> p 时趋近 100%)")
    print()

    # 不同 m 下的对比
    print(f"  {'m':<6} {'GPipe Peak':<15} {'1F1B Peak':<15} {'节省':<10}")
    print(f"  {'-'*46}")
    for m_val in [4, 8, 16, 32, 64]:
        gpipe_peak = m_val
        onefone_peak = p
        saving = (1 - onefone_peak / gpipe_peak) * 100
        print(f"  {m_val:<6} {gpipe_peak:<15} {onefone_peak:<15} {saving:.0f}%")


def main():
    p = 4   # stages
    m = 8   # micro-batches

    print("Pipeline Parallelism 调度可视化")
    print("  F = Forward (绿色), B = Backward (蓝色), ## = Bubble (灰色)")

    visualize_gpipe(p, m)
    visualize_1f1b(p, m)
    compare_memory(p, m)

    # 不同配置的 bubble 率
    print(f"\n{'='*70}")
    print(f"Bubble 率对比")
    print(f"{'='*70}")
    print(f"  {'Stages':<8} {'Micro-batches':<15} {'Bubble 率':<10}")
    print(f"  {'-'*33}")
    for p_val in [2, 4]:
        for m_val in [4, 8, 16, 32]:
            bubble = (p_val - 1) / (m_val + p_val - 1) * 100
            print(f"  {p_val:<8} {m_val:<15} {bubble:.1f}%")

    print(f"\n建议: m >= 4×p 使 bubble < 20%")


if __name__ == "__main__":
    main()
