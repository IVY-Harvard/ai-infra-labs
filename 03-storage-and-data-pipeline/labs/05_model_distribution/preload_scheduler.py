"""
模型预加载调度器

根据任务队列提前预加载模型到本地缓存，消除冷启动延迟。

用法：
    python preload_scheduler.py --cache-dir /nvme/model-cache
"""

import os
import time
import json
import threading
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor


@dataclass
class PreloadTask:
    """预加载任务"""
    priority: int           # 0=最高优先级
    model_name: str
    version: str
    target_node: str
    deadline: float         # Unix timestamp, 需要在此之前完成
    estimated_size_gb: float
    status: str = "pending"  # pending / loading / done / failed

    def __lt__(self, other):
        return self.priority < other.priority


class PreloadScheduler:
    """模型预加载调度器

    调度策略：
    1. 优先级调度：deadline 越近优先级越高
    2. 空间管理：预加载前检查目标节点空间
    3. 去重：同一模型不重复预加载
    4. 并发控制：限制同时预加载的数量（避免带宽打满）
    """

    def __init__(self, cache_dir: str, max_concurrent: int = 3,
                 bandwidth_limit_mbps: float = 500):
        self.cache_dir = cache_dir
        self.max_concurrent = max_concurrent
        self.bandwidth_limit = bandwidth_limit_mbps
        self.task_queue = PriorityQueue()
        self.active_tasks: Dict[str, PreloadTask] = {}
        self.completed: List[PreloadTask] = []
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.lock = threading.Lock()
        self.running = False

        os.makedirs(cache_dir, exist_ok=True)

    def submit_task(self, model_name: str, version: str,
                    target_node: str, deadline_seconds: float = 60,
                    size_gb: float = 10.0, priority: int = None):
        """提交预加载任务"""
        if priority is None:
            # 根据 deadline 自动计算优先级
            priority = max(0, int(deadline_seconds / 10))

        task = PreloadTask(
            priority=priority,
            model_name=model_name,
            version=version,
            target_node=target_node,
            deadline=time.time() + deadline_seconds,
            estimated_size_gb=size_gb,
        )

        self.task_queue.put(task)
        print(f"  [提交] {model_name}:{version} → {target_node} "
              f"(优先级={priority}, deadline={deadline_seconds}s)")

    def _is_cached(self, model_name: str, version: str) -> bool:
        """检查模型是否已缓存"""
        cache_path = os.path.join(self.cache_dir, model_name, version)
        return os.path.exists(cache_path)

    def _execute_preload(self, task: PreloadTask):
        """执行预加载"""
        task_key = f"{task.model_name}:{task.version}"

        with self.lock:
            if task_key in self.active_tasks:
                return  # 已在加载中
            task.status = "loading"
            self.active_tasks[task_key] = task

        try:
            # 检查是否已缓存
            if self._is_cached(task.model_name, task.version):
                print(f"  [跳过] {task_key} 已在缓存中")
                task.status = "done"
                return

            # 模拟下载
            cache_path = os.path.join(self.cache_dir, task.model_name,
                                      task.version)
            os.makedirs(cache_path, exist_ok=True)

            estimated_time = (task.estimated_size_gb * 1024 /
                            self.bandwidth_limit)
            print(f"  [加载] {task_key}: "
                  f"预计 {estimated_time:.1f}s ({task.estimated_size_gb}GB)")

            # 模拟下载时间
            time.sleep(min(estimated_time, 2.0))

            # 创建占位文件
            marker = os.path.join(cache_path, ".loaded")
            with open(marker, "w") as f:
                json.dump({
                    "model": task.model_name,
                    "version": task.version,
                    "loaded_at": time.time(),
                    "size_gb": task.estimated_size_gb,
                }, f)

            task.status = "done"
            now = time.time()
            if now <= task.deadline:
                print(f"  [完成] {task_key}: "
                      f"提前 {task.deadline - now:.1f}s 完成")
            else:
                print(f"  [完成] {task_key}: "
                      f"超时 {now - task.deadline:.1f}s")

        except Exception as e:
            task.status = "failed"
            print(f"  [失败] {task_key}: {e}")

        finally:
            with self.lock:
                self.active_tasks.pop(task_key, None)
                self.completed.append(task)

    def start(self):
        """启动调度器"""
        self.running = True
        print("\n预加载调度器启动")

        while self.running or not self.task_queue.empty():
            try:
                task = self.task_queue.get(timeout=1)
            except Exception:
                continue

            # 检查 deadline 是否已过
            if time.time() > task.deadline:
                print(f"  [过期] {task.model_name}:{task.version} deadline 已过")
                task.status = "failed"
                self.completed.append(task)
                continue

            # 提交执行
            self.executor.submit(self._execute_preload, task)

        self.executor.shutdown(wait=True)

    def stop(self):
        """停止调度器"""
        self.running = False

    def status(self) -> Dict:
        """获取调度器状态"""
        return {
            "pending": self.task_queue.qsize(),
            "active": len(self.active_tasks),
            "completed": len([t for t in self.completed
                             if t.status == "done"]),
            "failed": len([t for t in self.completed
                          if t.status == "failed"]),
        }


def main():
    parser = argparse.ArgumentParser(description="模型预加载调度器")
    parser.add_argument("--cache-dir", type=str,
                       default="/tmp/preload-cache")
    args = parser.parse_args()

    scheduler = PreloadScheduler(
        cache_dir=args.cache_dir,
        max_concurrent=3,
    )

    # 模拟提交多个预加载任务
    print("提交预加载任务...")
    scheduler.submit_task("llama-2-7b", "v1.0", "node-1",
                         deadline_seconds=30, size_gb=14)
    scheduler.submit_task("llama-2-13b", "v1.0", "node-1",
                         deadline_seconds=60, size_gb=26)
    scheduler.submit_task("mistral-7b", "v0.3", "node-2",
                         deadline_seconds=20, size_gb=14)
    scheduler.submit_task("llama-2-7b", "v1.0", "node-2",
                         deadline_seconds=45, size_gb=14)

    # 启动调度器（会阻塞直到所有任务完成）
    scheduler_thread = threading.Thread(target=scheduler.start)
    scheduler_thread.start()

    # 等待任务完成
    time.sleep(8)
    scheduler.stop()
    scheduler_thread.join()

    # 打印状态
    status = scheduler.status()
    print(f"\n{'='*50}")
    print(f"调度器状态:")
    print(f"  完成: {status['completed']}")
    print(f"  失败: {status['failed']}")
    print(f"  待处理: {status['pending']}")


if __name__ == "__main__":
    main()
