# 07 - 微调生产化

## 从实验到生产的鸿沟

```
实验阶段:                         生产阶段:
├── 单次训练                       ├── 持续训练 + 自动化
├── 手动调参                       ├── 超参搜索 + 配置管理
├── 目测效果                       ├── 系统化评估 + 自动报告
├── 手动部署                       ├── CI/CD + 灰度发布
├── 不关心复现                     ├── 完全可复现
└── 一个人搞定                     └── 团队协作 + 权限管理
```

## 完整的微调生产流水线

```
数据管理 → 训练调度 → 模型评估 → 模型注册 → 部署上线 → 监控反馈
   │          │          │          │          │          │
数据版本    分布式训练    自动评测    版本管理    A/B测试    质量监控
数据验证    超参搜索     评估报告    审批流程    灰度发布    数据回流
质量过滤    Checkpoint   安全检查    模型卡片    回滚机制    持续改进
```

## 数据管理

### 数据版本化

```python
# 使用 DVC (Data Version Control) 管理数据版本
"""
# 初始化 DVC
dvc init
dvc remote add -d storage s3://my-bucket/dvc-storage

# 追踪数据文件
dvc add data/training_v1.jsonl
git add data/training_v1.jsonl.dvc
git commit -m "Add training data v1"

# 数据变更
dvc add data/training_v2.jsonl
git add data/training_v2.jsonl.dvc
git commit -m "Update training data v2"

# 切换版本
git checkout v1.0
dvc checkout
"""

# 或使用 HuggingFace Datasets 管理
from datasets import Dataset

class DataVersionManager:
    """数据版本管理"""
    
    def __init__(self, hub_repo="my-org/training-data"):
        self.repo = hub_repo
    
    def push_version(self, data, version_tag, description=""):
        dataset = Dataset.from_list(data)
        dataset.push_to_hub(
            self.repo,
            revision=version_tag,
            commit_message=f"v{version_tag}: {description}"
        )
    
    def load_version(self, version_tag):
        return Dataset.from_hub(self.repo, revision=version_tag)
    
    def compare_versions(self, v1, v2):
        d1 = self.load_version(v1)
        d2 = self.load_version(v2)
        return {
            "v1_size": len(d1), "v2_size": len(d2),
            "added": len(d2) - len(d1),
        }
```

### 数据验证

```python
from pydantic import BaseModel, validator
from typing import List, Optional

class ConversationTurn(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str
    
    @validator("role")
    def validate_role(cls, v):
        assert v in ("system", "user", "assistant"), f"Invalid role: {v}"
        return v
    
    @validator("content")
    def validate_content(cls, v):
        assert len(v.strip()) > 0, "Content cannot be empty"
        return v.strip()

class TrainingExample(BaseModel):
    messages: List[ConversationTurn]
    metadata: Optional[dict] = None
    
    @validator("messages")
    def validate_messages(cls, v):
        # 至少一个 user 和一个 assistant
        roles = [m.role for m in v]
        assert "user" in roles, "Must have at least one user message"
        assert "assistant" in roles, "Must have at least one assistant message"
        return v

def validate_dataset(data_path):
    """验证整个数据集"""
    errors = []
    valid_count = 0
    
    with open(data_path) as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                TrainingExample(**item)
                valid_count += 1
            except Exception as e:
                errors.append({"line": i, "error": str(e)})
    
    return {
        "total": valid_count + len(errors),
        "valid": valid_count,
        "errors": len(errors),
        "error_details": errors[:10],  # 只返回前 10 个
    }
```

## 训练配置管理

### 配置文件设计

```yaml
# configs/train_config.yaml
experiment:
  name: "qwen2-7b-customer-service-v3"
  description: "客服场景微调，第三版"
  tags: ["customer_service", "lora", "v3"]
  
model:
  name: "Qwen/Qwen2-7B"
  revision: "main"
  torch_dtype: "bfloat16"

method:
  type: "lora"
  lora:
    r: 64
    alpha: 128
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]
    dropout: 0.05

data:
  train_file: "data://training-data:v3.0/train.jsonl"
  eval_file: "data://training-data:v3.0/eval.jsonl"
  max_seq_length: 2048
  
training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2.0e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.03
  weight_decay: 0.01
  bf16: true
  gradient_checkpointing: true

distributed:
  strategy: "auto"  # auto, ddp, fsdp, deepspeed
  num_gpus: 1

evaluation:
  benchmarks: ["mmlu", "ceval", "custom_cs_eval"]
  eval_steps: 500
  
output:
  dir: "outputs/${experiment.name}/${now}"
  save_strategy: "steps"
  save_steps: 500
  save_total_limit: 3
  
tracking:
  wandb_project: "llm-finetune"
  wandb_run_name: "${experiment.name}"
```

### 超参搜索

```python
# 使用 Optuna 进行超参搜索
import optuna

def objective(trial):
    # 搜索空间
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    rank = trial.suggest_categorical("rank", [16, 32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    epochs = trial.suggest_int("epochs", 1, 5)
    
    # 训练
    config = create_config(lr=lr, rank=rank, batch_size=batch_size, epochs=epochs)
    metrics = train_and_evaluate(config)
    
    return metrics["eval_score"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print(f"Best params: {best_params}")
print(f"Best score: {study.best_value}")
```

## 模型版本管理

### 模型注册表

```python
class ModelRegistry:
    """模型版本管理"""
    
    def __init__(self, storage_path):
        self.storage = storage_path
        self.registry_file = os.path.join(storage_path, "registry.json")
    
    def register(self, model_path, metadata):
        """注册新模型版本"""
        version = self._next_version()
        entry = {
            "version": version,
            "path": model_path,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
            "status": "registered",  # registered → validated → approved → deployed
        }
        self._save_entry(entry)
        return version
    
    def promote(self, version, new_status):
        """提升模型状态"""
        entry = self._load_entry(version)
        valid_transitions = {
            "registered": "validated",
            "validated": "approved",
            "approved": "deployed",
        }
        assert valid_transitions.get(entry["status"]) == new_status
        entry["status"] = new_status
        self._save_entry(entry)
    
    def get_deployed(self):
        """获取当前部署的模型"""
        registry = self._load_registry()
        deployed = [e for e in registry if e["status"] == "deployed"]
        return deployed[-1] if deployed else None
    
    def rollback(self, to_version):
        """回滚到指定版本"""
        current = self.get_deployed()
        if current:
            current["status"] = "rolled_back"
            self._save_entry(current)
        target = self._load_entry(to_version)
        target["status"] = "deployed"
        self._save_entry(target)
```

### 模型卡片

```python
model_card = {
    "model_id": "qwen2-7b-cs-v3.1",
    "base_model": "Qwen/Qwen2-7B",
    "version": "3.1",
    "created_at": "2024-12-15T10:30:00",
    "created_by": "ml-team",
    
    "training": {
        "method": "LoRA (r=64, alpha=128)",
        "data_version": "v3.0",
        "data_size": 45000,
        "training_time": "6.5 hours",
        "hardware": "1x H20 96GB",
        "hyperparameters": {"lr": 2e-4, "epochs": 3, "batch_size": 4},
    },
    
    "evaluation": {
        "mmlu": 61.5,
        "ceval": 57.2,
        "custom_cs_accuracy": 82.3,
        "safety_refusal_rate": 97.5,
        "mt_bench_score": 7.8,
    },
    
    "changes_from_previous": [
        "新增 5K 条客服对话数据",
        "修复了售后场景的格式问题",
        "学习率从 3e-4 调整到 2e-4",
    ],
    
    "known_limitations": [
        "退款政策更新后需要重新微调",
        "处理英文客服请求效果较差",
    ],
    
    "approved_by": "tech-lead",
    "approved_at": "2024-12-15T14:00:00",
}
```

## A/B 测试

### A/B 测试框架

```python
import random
import hashlib

class ABTestManager:
    """A/B 测试管理器"""
    
    def __init__(self):
        self.experiments = {}
    
    def create_experiment(self, name, model_a, model_b, traffic_split=0.5):
        self.experiments[name] = {
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,
            "results_a": [],
            "results_b": [],
        }
    
    def route_request(self, experiment_name, user_id):
        """基于用户 ID 确定性路由"""
        exp = self.experiments[experiment_name]
        # 使用 hash 保证同一用户始终路由到同一模型
        hash_val = int(hashlib.md5(
            f"{experiment_name}:{user_id}".encode()
        ).hexdigest(), 16)
        
        if (hash_val % 100) / 100 < exp["traffic_split"]:
            return "model_a", exp["model_a"]
        else:
            return "model_b", exp["model_b"]
    
    def record_feedback(self, experiment_name, group, score):
        """记录用户反馈"""
        exp = self.experiments[experiment_name]
        exp[f"results_{group[-1]}"].append(score)
    
    def analyze(self, experiment_name):
        """统计分析"""
        exp = self.experiments[experiment_name]
        from scipy import stats
        
        a_scores = exp["results_a"]
        b_scores = exp["results_b"]
        
        t_stat, p_value = stats.ttest_ind(a_scores, b_scores)
        
        return {
            "model_a_mean": sum(a_scores) / len(a_scores),
            "model_b_mean": sum(b_scores) / len(b_scores),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "winner": "model_a" if sum(a_scores)/len(a_scores) > sum(b_scores)/len(b_scores) else "model_b",
        }
```

## 持续微调

### 持续学习策略

```python
continuous_finetuning = {
    "增量微调": {
        "方式": "在新数据上继续微调现有模型",
        "风险": "灾难性遗忘",
        "缓解": "混合旧数据（replay buffer）",
    },
    "定期全量重训": {
        "方式": "定期收集新数据，从 base model 重新微调",
        "优点": "避免遗忘，简单可靠",
        "缺点": "计算开销大",
    },
    "LoRA 热替换": {
        "方式": "训练新 LoRA adapter，在线替换",
        "优点": "快速、风险低（可回滚）",
        "缺点": "adapter 不能无限叠加",
    },
}
```

### 数据回流

```python
class DataFeedbackLoop:
    """用户反馈 → 训练数据的闭环"""
    
    def collect_feedback(self, interaction_logs):
        """从线上交互日志收集反馈"""
        positive_examples = []
        negative_examples = []
        
        for log in interaction_logs:
            if log["user_rating"] >= 4:
                positive_examples.append({
                    "prompt": log["user_query"],
                    "chosen": log["model_response"],
                })
            elif log["user_rating"] <= 2:
                negative_examples.append({
                    "prompt": log["user_query"],
                    "rejected": log["model_response"],
                })
        
        return positive_examples, negative_examples
    
    def build_preference_pairs(self, positive, negative):
        """构建偏好对用于 DPO"""
        pairs = []
        # 匹配相似 prompt 的好/坏回答
        for pos in positive:
            for neg in negative:
                if self.is_similar_prompt(pos["prompt"], neg["prompt"]):
                    pairs.append({
                        "prompt": pos["prompt"],
                        "chosen": pos["chosen"],
                        "rejected": neg["rejected"],
                    })
        return pairs
```

## LLaMA-Factory 工程化使用

### 为什么选择 LLaMA-Factory

```
LLaMA-Factory 优势：
1. 开箱即用：配置文件即可训练，无需写代码
2. 方法齐全：SFT, RLHF, DPO, ORPO 全支持
3. 模型广泛：LLaMA, Qwen, Mistral, ChatGLM 等
4. WebUI：可视化训练界面
5. 社区活跃：更新快，bug 修复及时
```

### LLaMA-Factory 生产化配置

```yaml
# LLaMA-Factory config for production
### model
model_name_or_path: Qwen/Qwen2-7B
quantization_bit: 4  # QLoRA

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 128

### dataset
dataset: my_custom_dataset
template: qwen
cutoff_len: 2048
preprocessing_num_workers: 16

### output
output_dir: outputs/qwen2-7b-lora
logging_steps: 10
save_steps: 500
overwrite_output_dir: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true

### eval
val_size: 0.05
per_device_eval_batch_size: 8
eval_strategy: steps
eval_steps: 500
```

## Unsloth 加速

```python
# Unsloth 可以实现 2-5x 的训练加速
# 核心原理：手写 CUDA kernel 优化 LoRA 前向/反向传播

# 生产化使用
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2-7B",
    max_seq_length=2048,
    dtype=None,  # auto detect
    load_in_4bit=True,  # QLoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,  # Unsloth 建议 dropout=0
    bias="none",
)

# 训练速度对比（7B QLoRA, 1x H20）
# 标准 HuggingFace: ~1500 tokens/sec
# Unsloth:          ~4500 tokens/sec (3x)
```

## 监控与告警

### 训练监控

```python
# W&B 集成
import wandb

wandb.init(
    project="llm-finetune",
    name="qwen2-7b-cs-v3",
    config=training_config,
    tags=["production", "customer_service"],
)

# 自定义监控指标
monitoring_metrics = {
    "训练指标": ["loss", "learning_rate", "grad_norm"],
    "资源指标": ["gpu_utilization", "memory_usage", "throughput"],
    "质量指标": ["eval_loss", "eval_accuracy", "benchmark_score"],
    "告警规则": {
        "loss_spike": "loss 突增 > 2x 平均值",
        "gpu_oom": "显存使用 > 95%",
        "gradient_explosion": "grad_norm > 10.0",
        "training_stuck": "loss 连续 100 steps 无变化",
    },
}
```

### 部署后监控

```python
class ProductionMonitor:
    """生产环境模型监控"""
    
    def monitor_quality(self, responses, interval="1h"):
        """定期质量抽检"""
        sample = random.sample(responses, min(100, len(responses)))
        
        metrics = {
            "avg_length": sum(len(r) for r in sample) / len(sample),
            "refusal_rate": sum(1 for r in sample if self.is_refusal(r)) / len(sample),
            "format_compliance": sum(1 for r in sample if self.check_format(r)) / len(sample),
        }
        
        # 与基线对比
        for key, value in metrics.items():
            if abs(value - self.baseline[key]) > self.thresholds[key]:
                self.alert(f"Metric {key} drifted: {value} vs baseline {self.baseline[key]}")
        
        return metrics
```

## 生产化检查清单

```
□ 数据管理
  □ 数据版本化（DVC 或 HF Hub）
  □ 数据验证脚本
  □ 数据质量报告

□ 训练流程
  □ 配置文件化（不在代码中硬编码）
  □ 可复现（固定 seed，记录所有参数）
  □ 分布式训练就绪
  □ Checkpoint 定期保存 + 断点恢复

□ 评估
  □ 自动化评测流水线
  □ 基线对比
  □ 安全性评估
  □ 人工评估流程

□ 部署
  □ 模型注册 + 版本管理
  □ A/B 测试框架
  □ 灰度发布流程
  □ 回滚机制

□ 监控
  □ 训练过程监控（W&B）
  □ 线上质量监控
  □ 告警机制
  □ 数据回流
```
