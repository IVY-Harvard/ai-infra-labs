"""
GPU 训练/推理健康检查探针

作为 K8s liveness/readiness probe 使用。
检测训练进程是否还在正常工作（而不仅仅是进程还活着）。

场景：
  - NCCL 通信死锁：进程存活但停止了训练
  - GPU 计算卡死：进程在等待 CUDA kernel 完成
  - 显存泄漏：逐渐 OOM

使用方式（在 Pod spec 中）：
    livenessProbe:
      exec:
        command: ["python", "health_check_probe.py", "--mode=liveness"]
      periodSeconds: 30
      failureThreshold: 3
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta


# 配置
HEARTBEAT_FILE = "/tmp/training_heartbeat"      # 训练循环定期更新的心跳文件
PROGRESS_FILE = "/tmp/training_progress.json"    # 训练进度文件
GPU_CHECK_TIMEOUT = 5                            # nvidia-smi 超时（秒）


def check_liveness() -> tuple[bool, str]:
    """
    Liveness 检查：训练进程是否还在工作？

    检查项：
    1. 心跳文件是否在最近 N 秒内更新
    2. GPU 是否响应（nvidia-smi）
    3. 训练进程是否存在

    Returns:
        (是否健康, 原因)
    """
    reasons = []

    # 检查 1: 心跳文件
    heartbeat_ok, heartbeat_msg = _check_heartbeat(max_age_seconds=120)
    if not heartbeat_ok:
        reasons.append(heartbeat_msg)

    # 检查 2: GPU 响应
    gpu_ok, gpu_msg = _check_gpu_responsive()
    if not gpu_ok:
        reasons.append(gpu_msg)

    # 检查 3: 训练进程存在
    process_ok, process_msg = _check_training_process()
    if not process_ok:
        reasons.append(process_msg)

    if reasons:
        return False, "; ".join(reasons)
    return True, "all checks passed"


def check_readiness() -> tuple[bool, str]:
    """
    Readiness 检查：训练/推理服务是否准备好接受工作？

    检查项：
    1. 模型是否加载完成
    2. GPU 显存是否分配正常
    3. 训练是否已开始（第一个 step 完成）

    Returns:
        (是否就绪, 原因)
    """
    reasons = []

    # 检查 1: 模型加载状态
    model_ok, model_msg = _check_model_loaded()
    if not model_ok:
        reasons.append(model_msg)

    # 检查 2: 训练进度
    progress_ok, progress_msg = _check_training_started()
    if not progress_ok:
        reasons.append(progress_msg)

    if reasons:
        return False, "; ".join(reasons)
    return True, "ready"


def check_startup() -> tuple[bool, str]:
    """
    Startup 检查：用于大模型场景，给予足够的启动时间。

    大模型加载可能需要 5-10 分钟，这个探针更宽松。
    """
    model_loaded_marker = Path("/tmp/model_loaded")
    if model_loaded_marker.exists():
        return True, "model loaded"
    return False, "model still loading"


def _check_heartbeat(max_age_seconds: int = 120) -> tuple[bool, str]:
    """检查心跳文件"""
    heartbeat_path = Path(HEARTBEAT_FILE)

    if not heartbeat_path.exists():
        # 如果心跳文件不存在，可能还在初始化
        return True, "no heartbeat file (initializing?)"

    try:
        mtime = heartbeat_path.stat().st_mtime
        age = time.time() - mtime

        if age > max_age_seconds:
            return False, (
                f"heartbeat stale: last update {age:.0f}s ago "
                f"(threshold: {max_age_seconds}s)"
            )
        return True, f"heartbeat fresh ({age:.0f}s ago)"

    except Exception as e:
        return False, f"heartbeat check error: {e}"


def _check_gpu_responsive() -> tuple[bool, str]:
    """检查 GPU 是否响应"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=GPU_CHECK_TIMEOUT,
        )
        if result.returncode != 0:
            return False, f"nvidia-smi failed: {result.stderr.strip()}"
        return True, "GPU responsive"
    except subprocess.TimeoutExpired:
        return False, f"nvidia-smi timeout ({GPU_CHECK_TIMEOUT}s)"
    except FileNotFoundError:
        # 可能在没有 GPU 的环境运行测试
        return True, "nvidia-smi not found (non-GPU env?)"


def _check_training_process() -> tuple[bool, str]:
    """检查训练进程是否存在"""
    pid_file = Path("/tmp/trainer.pid")
    if not pid_file.exists():
        return True, "no PID file (may be initializing)"

    try:
        pid = int(pid_file.read_text().strip())
        # 检查进程是否存在（发送信号 0）
        os.kill(pid, 0)
        return True, f"training process alive (PID={pid})"
    except ProcessLookupError:
        return False, f"training process dead (PID={pid})"
    except ValueError:
        return False, "invalid PID file"


def _check_model_loaded() -> tuple[bool, str]:
    """检查模型是否加载完成"""
    marker = Path("/tmp/model_loaded")
    if marker.exists():
        return True, "model loaded"
    return False, "model not loaded yet"


def _check_training_started() -> tuple[bool, str]:
    """检查训练是否已开始"""
    progress_path = Path(PROGRESS_FILE)
    if not progress_path.exists():
        return False, "no progress file"

    try:
        progress = json.loads(progress_path.read_text())
        step = progress.get("current_step", 0)
        if step > 0:
            return True, f"training at step {step}"
        return False, "training not started (step=0)"
    except (json.JSONDecodeError, Exception) as e:
        return False, f"progress file error: {e}"


def main():
    parser = argparse.ArgumentParser(description="GPU 健康检查探针")
    parser.add_argument(
        "--mode",
        choices=["liveness", "readiness", "startup"],
        required=True,
        help="探针模式",
    )
    args = parser.parse_args()

    check_func = {
        "liveness": check_liveness,
        "readiness": check_readiness,
        "startup": check_startup,
    }[args.mode]

    healthy, reason = check_func()

    if healthy:
        print(f"OK: {reason}")
        sys.exit(0)
    else:
        print(f"FAIL: {reason}", file=sys.stderr)
        sys.exit(1)


# --- 训练代码中的心跳集成示例 ---

class TrainingHeartbeat:
    """
    在训练代码中使用，定期更新心跳文件。

    使用方式：
        heartbeat = TrainingHeartbeat()
        for step in range(total_steps):
            train_step()
            heartbeat.update(step=step)
    """

    def __init__(
        self,
        heartbeat_file: str = HEARTBEAT_FILE,
        progress_file: str = PROGRESS_FILE,
    ):
        self.heartbeat_file = Path(heartbeat_file)
        self.progress_file = Path(progress_file)

    def update(self, step: int, loss: float = 0.0, **kwargs):
        """更新心跳和进度"""
        # 更新心跳（touch 文件）
        self.heartbeat_file.touch()

        # 更新进度信息
        progress = {
            "current_step": step,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.progress_file.write_text(json.dumps(progress))

    def mark_model_loaded(self):
        """标记模型加载完成"""
        Path("/tmp/model_loaded").touch()

    def write_pid(self):
        """写入当前进程 PID"""
        Path("/tmp/trainer.pid").write_text(str(os.getpid()))


if __name__ == "__main__":
    main()
