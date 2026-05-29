"""
部署管理器
"""

import os
import json
import subprocess
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class DeployConfig:
    """部署配置"""
    model_path: str
    deploy_name: str
    engine: str = "vllm"  # vllm, tgi
    tensor_parallel: int = 1
    port: int = 8000
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9


class DeployManager:
    """部署管理器"""

    def __init__(self, registry_path: str = "./deploy_registry"):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)
        self.registry_file = os.path.join(registry_path, "deploys.json")
        self.deploys = self._load_registry()

    def _load_registry(self) -> List[Dict]:
        if os.path.exists(self.registry_file):
            with open(self.registry_file) as f:
                return json.load(f)
        return []

    def _save_registry(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.deploys, f, indent=2)

    def deploy(self, config: DeployConfig) -> Dict:
        """部署模型"""
        print(f"部署模型: {config.deploy_name}")
        print(f"  引擎: {config.engine}")
        print(f"  路径: {config.model_path}")
        print(f"  TP: {config.tensor_parallel}")

        deploy_cmd = self._build_command(config)
        print(f"  命令: {deploy_cmd}")

        # 记录部署信息
        deploy_info = {
            "name": config.deploy_name,
            "model_path": config.model_path,
            "engine": config.engine,
            "port": config.port,
            "tensor_parallel": config.tensor_parallel,
            "status": "deployed",
            "command": deploy_cmd,
            "deployed_at": datetime.now().isoformat(),
        }

        self.deploys.append(deploy_info)
        self._save_registry()

        return deploy_info

    def _build_command(self, config: DeployConfig) -> str:
        """构建部署命令"""
        if config.engine == "vllm":
            cmd = (
                f"vllm serve {config.model_path} "
                f"--port {config.port} "
                f"--tensor-parallel-size {config.tensor_parallel} "
                f"--max-model-len {config.max_model_len} "
                f"--gpu-memory-utilization {config.gpu_memory_utilization}"
            )
        elif config.engine == "tgi":
            cmd = (
                f"text-generation-launcher "
                f"--model-id {config.model_path} "
                f"--port {config.port} "
                f"--num-shard {config.tensor_parallel}"
            )
        else:
            cmd = f"echo 'Unsupported engine: {config.engine}'"

        return cmd

    def list_deployments(self) -> List[Dict]:
        """列出所有部署"""
        return self.deploys

    def stop(self, deploy_name: str):
        """停止部署"""
        for deploy in self.deploys:
            if deploy["name"] == deploy_name:
                deploy["status"] = "stopped"
                deploy["stopped_at"] = datetime.now().isoformat()
        self._save_registry()
        print(f"已停止: {deploy_name}")

    def get_active_deployment(self) -> Optional[Dict]:
        """获取当前活跃的部署"""
        active = [d for d in self.deploys if d["status"] == "deployed"]
        return active[-1] if active else None
