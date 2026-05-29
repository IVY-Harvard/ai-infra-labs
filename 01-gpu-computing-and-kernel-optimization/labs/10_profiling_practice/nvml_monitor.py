"""
Lab 10: NVML 实时 GPU 监控

使用 pynvml 实时监控 GPU 状态，类似 nvidia-smi 但更灵活。
适合在运行推理/训练时在旁边开一个终端监控。

Usage:
    python nvml_monitor.py              # 监控所有 GPU
    python nvml_monitor.py --gpu 0      # 只监控 GPU 0
    python nvml_monitor.py --interval 2 # 每 2 秒刷新
    python nvml_monitor.py --csv        # 输出 CSV 格式（用于后续分析）
"""

import argparse
import time
import sys
from datetime import datetime

try:
    import pynvml
except ImportError:
    print("请安装 pynvml: pip install pynvml")
    sys.exit(1)


class GPUMonitor:
    """GPU 实时监控器"""

    def __init__(self, gpu_ids=None):
        pynvml.nvmlInit()
        self.driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(self.driver_version, bytes):
            self.driver_version = self.driver_version.decode('utf-8')

        total_gpus = pynvml.nvmlDeviceGetCount()
        self.gpu_ids = gpu_ids if gpu_ids else list(range(total_gpus))
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_ids]

        print(f"NVML 初始化: Driver {self.driver_version}, 监控 GPU: {self.gpu_ids}")

    def get_status(self, handle):
        """获取单个 GPU 的状态"""
        status = {}

        # 基本信息
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        status['name'] = name

        # 利用率
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            status['gpu_util'] = util.gpu
            status['mem_util'] = util.memory
        except Exception:
            status['gpu_util'] = -1
            status['mem_util'] = -1

        # 内存
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        status['mem_used_gb'] = mem.used / (1024**3)
        status['mem_total_gb'] = mem.total / (1024**3)
        status['mem_used_pct'] = mem.used / mem.total * 100

        # 温度
        try:
            status['temp'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            status['temp'] = -1

        # 功耗
        try:
            status['power_w'] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            status['power_limit_w'] = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except Exception:
            status['power_w'] = -1
            status['power_limit_w'] = -1

        # 频率
        try:
            status['sm_clock'] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            status['mem_clock'] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except Exception:
            status['sm_clock'] = -1
            status['mem_clock'] = -1

        # 进程
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            status['num_processes'] = len(procs)
        except Exception:
            status['num_processes'] = -1

        return status

    def print_header(self):
        """打印表头"""
        print("\n" + "=" * 100)
        print(f"{'GPU':>4} | {'Name':>12} | {'Util%':>6} | {'Mem%':>5} | "
              f"{'Mem Used':>9} | {'Temp':>5} | {'Power':>10} | "
              f"{'SM MHz':>7} | {'Procs':>5}")
        print("-" * 100)

    def print_status(self, gpu_id, status):
        """打印一行状态"""
        power_str = f"{status['power_w']:.0f}/{status['power_limit_w']:.0f}W"
        mem_str = f"{status['mem_used_gb']:.1f}/{status['mem_total_gb']:.0f}GB"

        # 颜色标记（高利用率/温度用醒目色）
        gpu_util = status['gpu_util']
        temp = status['temp']

        print(f"{gpu_id:>4} | {status['name'][:12]:>12} | {gpu_util:>5}% | "
              f"{status['mem_used_pct']:>4.0f}% | "
              f"{mem_str:>9} | {temp:>4}°C | {power_str:>10} | "
              f"{status['sm_clock']:>6} | {status['num_processes']:>5}")

    def print_csv_header(self):
        """CSV 表头"""
        fields = ['timestamp', 'gpu_id', 'gpu_util', 'mem_util', 'mem_used_gb',
                  'temp', 'power_w', 'sm_clock']
        print(','.join(fields))

    def print_csv_row(self, gpu_id, status):
        """CSV 数据行"""
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [ts, str(gpu_id), str(status['gpu_util']),
               str(status['mem_util']), f"{status['mem_used_gb']:.2f}",
               str(status['temp']), f"{status['power_w']:.1f}",
               str(status['sm_clock'])]
        print(','.join(row))

    def monitor_once(self, csv=False):
        """采集一次数据"""
        if not csv:
            self.print_header()

        for idx, (gpu_id, handle) in enumerate(zip(self.gpu_ids, self.handles)):
            status = self.get_status(handle)
            if csv:
                self.print_csv_row(gpu_id, status)
            else:
                self.print_status(gpu_id, status)

    def monitor_loop(self, interval=1.0, csv=False, duration=None):
        """持续监控循环"""
        if csv:
            self.print_csv_header()

        start_time = time.time()
        try:
            while True:
                if not csv:
                    # 清屏效果（打印足够多的换行）
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                          f"GPU Monitor (Ctrl+C 退出, 间隔 {interval}s)")

                self.monitor_once(csv)

                if duration and (time.time() - start_time) > duration:
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n监控结束")
        finally:
            pynvml.nvmlShutdown()

    def snapshot_summary(self):
        """一次性摘要（不循环）"""
        print("\n" + "=" * 60)
        print("GPU 状态摘要")
        print("=" * 60)

        total_power = 0
        total_mem_used = 0

        for gpu_id, handle in zip(self.gpu_ids, self.handles):
            status = self.get_status(handle)

            print(f"\nGPU {gpu_id}: {status['name']}")
            print(f"  利用率: GPU {status['gpu_util']}%, Memory {status['mem_util']}%")
            print(f"  显存: {status['mem_used_gb']:.1f} / {status['mem_total_gb']:.0f} GB "
                  f"({status['mem_used_pct']:.0f}%)")
            print(f"  温度: {status['temp']}°C")
            print(f"  功耗: {status['power_w']:.0f} / {status['power_limit_w']:.0f} W")
            print(f"  频率: SM {status['sm_clock']} MHz, Mem {status['mem_clock']} MHz")
            print(f"  计算进程数: {status['num_processes']}")

            total_power += status['power_w']
            total_mem_used += status['mem_used_gb']

            # 警告
            if status['temp'] > 80:
                print(f"  ⚠ 温度偏高，可能触发降频")
            if status['gpu_util'] > 0 and status['gpu_util'] < 30:
                print(f"  ⚠ GPU 利用率低，可能存在瓶颈")
            if status['power_w'] > status['power_limit_w'] * 0.95:
                print(f"  ⚠ 接近功耗上限")

        print(f"\n总计:")
        print(f"  总功耗: {total_power:.0f} W")
        print(f"  总显存使用: {total_mem_used:.1f} GB")

        pynvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description='GPU 实时监控')
    parser.add_argument('--gpu', type=int, nargs='*', default=None,
                       help='要监控的 GPU ID（默认全部）')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='刷新间隔（秒）')
    parser.add_argument('--csv', action='store_true',
                       help='输出 CSV 格式')
    parser.add_argument('--once', action='store_true',
                       help='只采集一次（不循环）')
    parser.add_argument('--duration', type=float, default=None,
                       help='监控时长（秒）')

    args = parser.parse_args()

    monitor = GPUMonitor(gpu_ids=args.gpu)

    if args.once:
        monitor.snapshot_summary()
    else:
        monitor.monitor_loop(
            interval=args.interval,
            csv=args.csv,
            duration=args.duration,
        )


if __name__ == "__main__":
    main()
