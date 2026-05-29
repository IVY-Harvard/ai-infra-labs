"""
Spot 实例中断处理器

监听云服务商的 Spot 中断通知，在实例被回收前：
1. 触发 checkpoint 保存
2. 通知分布式训练的其他 worker
3. 优雅终止训练进程

支持的云服务商：
  - AWS: Instance Metadata Service 中断通知
  - GCP: Preemptible VM metadata
  - Azure: Scheduled Events

使用方式：
    python spot_handler.py --provider=aws --checkpoint-dir=/checkpoints
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading
import subprocess
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

try:
    import requests
except ImportError:
    requests = None  # 在没有 requests 库的环境中用 urllib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("spot-handler")


class SpotInterruptionHandler:
    """Spot 中断处理器基类"""

    def __init__(
        self,
        checkpoint_dir: str = "/checkpoints",
        checkpoint_cmd: Optional[str] = None,
        graceful_period_sec: int = 25,  # Spot 通常给 30s，留 5s 余量
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_cmd = checkpoint_cmd
        self.graceful_period_sec = graceful_period_sec
        self._interrupted = threading.Event()
        self._callbacks: list[Callable] = []

    def register_callback(self, callback: Callable):
        """注册中断回调函数"""
        self._callbacks.append(callback)

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()

    def handle_interruption(self):
        """处理中断"""
        logger.warning("=" * 60)
        logger.warning("SPOT 中断信号收到！开始优雅退出...")
        logger.warning("=" * 60)
        self._interrupted.set()

        # 1. 触发 checkpoint
        self._save_checkpoint()

        # 2. 执行注册的回调
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"回调执行失败: {e}")

        # 3. 通知训练进程
        self._notify_training_process()

        logger.info("优雅退出完成")

    def _save_checkpoint(self):
        """触发 checkpoint 保存"""
        if self.checkpoint_cmd:
            logger.info(f"执行 checkpoint 命令: {self.checkpoint_cmd}")
            try:
                result = subprocess.run(
                    self.checkpoint_cmd,
                    shell=True,
                    timeout=self.graceful_period_sec - 5,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    logger.info("Checkpoint 保存成功")
                else:
                    logger.error(f"Checkpoint 保存失败: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.error("Checkpoint 保存超时！")
        else:
            # 通过信号通知训练进程保存 checkpoint
            self._signal_checkpoint()

    def _signal_checkpoint(self):
        """通过 SIGUSR1 通知训练进程保存 checkpoint"""
        pid_file = self.checkpoint_dir / "trainer.pid"
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            logger.info(f"发送 SIGUSR1 到训练进程 PID={pid}")
            try:
                os.kill(pid, signal.SIGUSR1)
                # 等待 checkpoint 完成
                ckpt_done = self.checkpoint_dir / "checkpoint_done"
                for _ in range(self.graceful_period_sec - 5):
                    if ckpt_done.exists():
                        logger.info("训练进程已完成 checkpoint")
                        ckpt_done.unlink()
                        return
                    time.sleep(1)
                logger.warning("等待 checkpoint 超时")
            except ProcessLookupError:
                logger.warning(f"训练进程 PID={pid} 不存在")

    def _notify_training_process(self):
        """通知训练进程终止"""
        pid_file = self.checkpoint_dir / "trainer.pid"
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            logger.info(f"发送 SIGTERM 到训练进程 PID={pid}")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


class AWSSpotHandler(SpotInterruptionHandler):
    """AWS Spot 中断处理"""

    METADATA_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"
    TOKEN_URL = "http://169.254.169.254/latest/api/token"

    def poll(self, interval: int = 5):
        """轮询 AWS Instance Metadata 中断通知"""
        logger.info("开始监听 AWS Spot 中断通知...")

        while not self.interrupted:
            try:
                # 获取 IMDSv2 token
                token_resp = requests.put(
                    self.TOKEN_URL,
                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                    timeout=2,
                )
                token = token_resp.text

                # 检查中断通知
                resp = requests.get(
                    self.METADATA_URL,
                    headers={"X-aws-ec2-metadata-token": token},
                    timeout=2,
                )

                if resp.status_code == 200:
                    # 中断通知！
                    action = resp.json()
                    logger.warning(f"AWS Spot 中断: action={action}")
                    self.handle_interruption()
                    return

            except requests.exceptions.RequestException:
                pass  # 正常情况：没有中断

            time.sleep(interval)


class GCPSpotHandler(SpotInterruptionHandler):
    """GCP Preemptible VM 中断处理"""

    METADATA_URL = (
        "http://metadata.google.internal/computeMetadata/v1/"
        "instance/preempted"
    )

    def poll(self, interval: int = 5):
        """轮询 GCP metadata 中断通知"""
        logger.info("开始监听 GCP Preemptible 中断通知...")

        while not self.interrupted:
            try:
                resp = requests.get(
                    self.METADATA_URL,
                    headers={"Metadata-Flavor": "Google"},
                    timeout=2,
                )
                if resp.status_code == 200 and resp.text.strip() == "TRUE":
                    logger.warning("GCP Preemptible VM 即将被回收！")
                    self.handle_interruption()
                    return
            except requests.exceptions.RequestException:
                pass

            time.sleep(interval)


class K8sNodeTerminationHandler(SpotInterruptionHandler):
    """
    通用 K8s 节点终止处理器。
    监听 Pod 的 SIGTERM 信号（K8s 驱逐时发送）。
    """

    def setup(self):
        """注册信号处理"""
        logger.info("注册 SIGTERM 处理器...")
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigterm_handler)

    def _sigterm_handler(self, signum, frame):
        logger.warning(f"收到信号 {signum}，开始处理中断...")
        self.handle_interruption()

    def wait(self):
        """阻塞等待信号"""
        logger.info("等待终止信号...")
        while not self.interrupted:
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Spot 实例中断处理器")
    parser.add_argument(
        "--provider",
        choices=["aws", "gcp", "k8s"],
        default="k8s",
        help="云服务商",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/checkpoints",
        help="Checkpoint 目录",
    )
    parser.add_argument(
        "--checkpoint-cmd",
        default=None,
        help="Checkpoint 保存命令",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="轮询间隔（秒）",
    )
    parser.add_argument(
        "--graceful-period",
        type=int,
        default=25,
        help="优雅退出时间（秒）",
    )
    args = parser.parse_args()

    handler_cls = {
        "aws": AWSSpotHandler,
        "gcp": GCPSpotHandler,
        "k8s": K8sNodeTerminationHandler,
    }[args.provider]

    handler = handler_cls(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_cmd=args.checkpoint_cmd,
        graceful_period_sec=args.graceful_period,
    )

    # 注册额外的清理回调
    handler.register_callback(
        lambda: logger.info("写入中断标记文件...")
    )

    if args.provider == "k8s":
        handler.setup()
        handler.wait()
    else:
        handler.poll(interval=args.poll_interval)

    logger.info("Spot Handler 退出")
    sys.exit(0)


if __name__ == "__main__":
    main()
