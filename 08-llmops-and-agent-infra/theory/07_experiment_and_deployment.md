# 实验到部署：从开发到生产的完整流水线

## 1. 概述

### 1.1 LLM 应用的发布挑战

传统微服务发布的确定性 vs LLM 应用的不确定性：

```
传统微服务：
  代码变更 → 单元测试 → 集成测试 → 灰度 → 全量
  特点：确定性强，测试覆盖后可高置信发布

LLM 应用：
  Prompt/RAG/模型变更 → 离线评估 → 灰度 → 在线评估 → 全量
  特点：非确定性，即使评估通过也可能在特定场景出问题
  需要：更保守的灰度策略 + 持续在线评估 + 快速回滚
```

### 1.2 发布对象

LLM 应用中可能发生变更的组件：

| 变更对象 | 风险等级 | 发布策略 |
|----------|----------|----------|
| Prompt 文本 | 中 | A/B 测试 + 灰度 |
| RAG 配置（chunk_size/top_k） | 中 | 影子测试 + 灰度 |
| Embedding 模型 | 高 | 全量重新索引 + 蓝绿切换 |
| LLM 模型版本 | 高 | 影子测试 + 长期灰度 |
| Agent 编排逻辑 | 高 | 影子测试 + 逐步灰度 |
| 工具/插件 | 低 | 功能开关 + 灰度 |

## 2. 实验追踪

### 2.1 MLflow

```
定位：开源 ML 生命周期管理平台
核心组件：
  ├── Tracking：实验记录
  ├── Projects：可复现运行
  ├── Models：模型打包
  └── Registry：模型注册中心

在 LLMOps 中的角色：
  • 记录每次 Prompt/RAG 配置变更的实验结果
  • 管理评估数据集版本
  • 模型注册与版本管理
  • 部署触发
```

```python
import mlflow

# 记录一次 RAG 实验
with mlflow.start_run(run_name="rag_v2_hyde_reranker"):
    # 记录参数
    mlflow.log_params({
        "prompt_version": "v2.1",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": "bge-m3",
        "reranker": "bge-reranker-v2-m3",
        "top_k": 5,
        "hyde_enabled": True,
        "llm_model": "qwen2.5-72b",
        "temperature": 0.3,
    })
    
    # 运行评估
    eval_results = run_evaluation(eval_dataset)
    
    # 记录指标
    mlflow.log_metrics({
        "faithfulness": eval_results.faithfulness,
        "relevancy": eval_results.relevancy,
        "correctness": eval_results.correctness,
        "latency_p50": eval_results.latency_p50,
        "latency_p95": eval_results.latency_p95,
        "cost_per_query": eval_results.cost_per_query,
    })
    
    # 记录配置文件
    mlflow.log_artifact("prompts/qa/v2.1.yaml")
    mlflow.log_artifact("configs/rag_config.yaml")
```

### 2.2 Weights & Biases (W&B)

```
定位：实验追踪 + 可视化 + 协作平台
优势：
  • 可视化能力强（对比多次实验）
  • 团队协作特性
  • Sweeps（超参搜索）
  • Tables（结构化数据追踪）
  • Prompts（Prompt 追踪）
```

```python
import wandb

# 初始化实验
run = wandb.init(
    project="rag-optimization",
    name="hyde_reranker_experiment",
    config={
        "prompt_version": "v2.1",
        "chunk_size": 1000,
        "embedding_model": "bge-m3",
        "reranker_model": "bge-reranker-v2-m3",
    }
)

# 记录评估结果（支持表格）
eval_table = wandb.Table(columns=["question", "answer", "score", "latency"])
for result in eval_results:
    eval_table.add_data(
        result.question, result.answer,
        result.score, result.latency
    )
wandb.log({"evaluation": eval_table})

# 记录聚合指标
wandb.log({
    "faithfulness": 0.92,
    "relevancy": 0.88,
    "avg_latency_ms": 1500,
})
```

### 2.3 实验对比

```
MLflow vs W&B 在 LLMOps 中的选择：

┌──────────────┬──────────────────┬──────────────────┐
│  维度        │  MLflow           │  W&B             │
├──────────────┼──────────────────┼──────────────────┤
│  部署方式    │  自托管           │  Cloud / 自托管  │
│  开源        │  完全开源         │  部分开源        │
│  模型注册    │  内置             │  有限            │
│  可视化      │  基础             │  强大            │
│  LLM 追踪   │  mlflow.llm       │  W&B Prompts     │
│  团队协作    │  基础             │  强大            │
│  CI/CD 集成  │  好               │  好              │
│  推荐场景    │  自托管 + 模型管理│  团队协作 + 可视化│
└──────────────┴──────────────────┴──────────────────┘
```

## 3. 模型注册中心

### 3.1 概念

```
模型注册中心 = LLM 应用的"制品仓库"

管理对象：
├── Prompt 版本
├── RAG 配置版本
├── Agent 编排配置
├── Embedding 模型
├── Fine-tuned 模型权重
└── 评估数据集版本

生命周期：
  Staging → Canary → Production → Archived
```

### 3.2 实现

```python
class ModelRegistry:
    """LLM 应用配置注册中心"""
    
    def __init__(self, backend="mlflow"):
        self.backend = backend
    
    def register_version(self, app_name: str, version: str,
                         config: dict, eval_results: dict):
        """注册新版本"""
        # 验证评估结果达标
        if not self._passes_quality_gate(eval_results):
            raise ValueError("评估未通过质量门禁")
        
        self.backend.register(
            name=app_name,
            version=version,
            config=config,
            metadata={
                "eval_results": eval_results,
                "registered_at": datetime.now(),
                "stage": "staging",
            }
        )
    
    def promote(self, app_name: str, version: str,
                target_stage: str):
        """提升版本阶段"""
        valid_transitions = {
            "staging": ["canary"],
            "canary": ["production", "archived"],
            "production": ["archived"],
        }
        current_stage = self._get_stage(app_name, version)
        if target_stage not in valid_transitions[current_stage]:
            raise ValueError(f"非法状态转换: {current_stage} → {target_stage}")
        
        self.backend.set_stage(app_name, version, target_stage)
```

## 4. 灰度发布

### 4.1 策略

```
┌─────────────────────────────────────────────────────┐
│              灰度发布策略                             │
│                                                       │
│  策略 1：基于流量比例                                 │
│  ├── 1% → 5% → 10% → 25% → 50% → 100%             │
│  ├── 每阶段持续观察 1-24h                            │
│  └── 质量指标异常自动回滚                            │
│                                                       │
│  策略 2：基于用户分组                                 │
│  ├── 内部用户 → Beta 用户 → 新用户 → 全量           │
│  ├── 高价值用户最后切换                              │
│  └── 按组收集反馈                                    │
│                                                       │
│  策略 3：基于场景                                     │
│  ├── 低风险场景 → 中风险场景 → 高风险场景           │
│  ├── 例：闲聊 → 知识问答 → 交易操作                 │
│  └── 逐步扩大场景覆盖                                │
└─────────────────────────────────────────────────────┘
```

### 4.2 灰度决策引擎

```python
class CanaryDecisionEngine:
    """灰度决策引擎 - 自动化灰度推进"""
    
    def __init__(self, config: CanaryConfig):
        self.config = config
        self.stages = config.stages  # [1%, 5%, 10%, 25%, 50%, 100%]
        self.current_stage = 0
    
    async def evaluate_stage(self) -> str:
        """评估当前阶段是否可以推进"""
        metrics = await self._collect_metrics()
        
        # 关键指标检查
        checks = {
            "quality": metrics.quality_score >= self.config.min_quality,
            "latency": metrics.p95_latency <= self.config.max_latency,
            "error_rate": metrics.error_rate <= self.config.max_error_rate,
            "cost": metrics.cost_per_query <= self.config.max_cost,
            "duration": metrics.stage_duration >= self.config.min_stage_duration,
        }
        
        if all(checks.values()):
            return "promote"    # 推进到下一阶段
        elif any(self._is_critical_failure(k, v) for k, v in checks.items()):
            return "rollback"   # 关键指标严重劣化，回滚
        else:
            return "hold"       # 继续观察
    
    async def advance(self):
        decision = await self.evaluate_stage()
        if decision == "promote":
            self.current_stage += 1
            await self._update_traffic_ratio(
                self.stages[self.current_stage]
            )
        elif decision == "rollback":
            await self._rollback()
```

## 5. 影子测试（Shadow Testing）

### 5.1 原理

```
┌─────────────────────────────────────────────────────┐
│              影子测试架构                             │
│                                                       │
│               ┌──────────────┐                       │
│               │   用户请求    │                       │
│               └──────┬───────┘                       │
│                      │                               │
│               ┌──────▼───────┐                       │
│               │   流量复制    │                       │
│               └──────┬───────┘                       │
│            ┌─────────┼─────────┐                     │
│            ▼                   ▼                     │
│     ┌──────────────┐   ┌──────────────┐            │
│     │  生产版本     │   │  影子版本     │            │
│     │  (v1.0)      │   │  (v2.0)      │            │
│     └──────┬───────┘   └──────┬───────┘            │
│            │                   │                     │
│            ▼                   ▼                     │
│     ┌──────────────┐   ┌──────────────┐            │
│     │ 返回给用户    │   │  仅记录结果   │            │
│     └──────────────┘   │  不返回用户   │            │
│                         └──────┬───────┘            │
│                                ▼                     │
│                         ┌──────────────┐            │
│                         │  结果对比     │            │
│                         │  质量评估     │            │
│                         └──────────────┘            │
└─────────────────────────────────────────────────────┘

优势：
  • 零用户影响（用户始终看到生产版本的结果）
  • 真实流量评估（比离线评估更接近真实场景）
  • 可长期运行观察

劣势：
  • 双倍资源消耗
  • 有副作用的工具调用需要 mock
  • 异步结果对比有延迟
```

### 5.2 影子测试实现

```python
class ShadowTester:
    """影子测试框架"""
    
    def __init__(self, production_app, shadow_app, comparator):
        self.production = production_app
        self.shadow = shadow_app
        self.comparator = comparator
    
    async def handle_request(self, request):
        # 生产版本正常处理
        prod_task = asyncio.create_task(
            self.production.process(request)
        )
        
        # 影子版本异步处理（不阻塞响应）
        shadow_task = asyncio.create_task(
            self._shadow_process(request)
        )
        
        # 返回生产结果给用户
        prod_result = await prod_task
        return prod_result
    
    async def _shadow_process(self, request):
        """影子版本处理（异步，不影响用户）"""
        try:
            shadow_result = await self.shadow.process(request)
            
            # 等待生产结果完成后对比
            prod_result = await self.production.get_cached_result(request.id)
            
            comparison = self.comparator.compare(
                prod_result, shadow_result
            )
            
            # 记录对比结果
            await self._log_comparison(request, comparison)
            
        except Exception as e:
            # 影子测试失败不影响生产
            logger.warning(f"Shadow test failed: {e}")
```

## 6. 在线 A/B 测试

### 6.1 与 Prompt A/B 的区别

```
Prompt A/B 测试：仅测试 Prompt 变量
在线 A/B 测试：测试整体系统变更（模型/RAG/Agent 配置）

在线 A/B 测试设计要点：
├── 用户粘性：同一用户在测试期间始终看到同一版本
├── 分流均匀：确保实验组和对照组用户分布均匀
├── 指标设计：业务指标 + 质量指标 + 效率指标
├── 样本量：确保统计显著性所需的最小样本
└── 长期效果：某些变更的效果需要较长时间显现
```

### 6.2 A/B 测试平台

```python
class ABTestPlatform:
    """在线 A/B 测试平台"""
    
    def __init__(self):
        self.experiments = {}
    
    def create_experiment(self, name: str, variants: list[dict],
                          traffic_allocation: dict):
        """
        创建实验
        variants: [
            {"name": "control", "config": {...}},
            {"name": "treatment_a", "config": {...}},
        ]
        traffic_allocation: {"control": 0.8, "treatment_a": 0.2}
        """
        self.experiments[name] = Experiment(
            name=name,
            variants=variants,
            allocation=traffic_allocation,
            start_time=datetime.now(),
        )
    
    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """为用户确定分组（确定性哈希，保证粘性）"""
        experiment = self.experiments[experiment_name]
        hash_val = hashlib.md5(
            f"{experiment_name}:{user_id}".encode()
        ).hexdigest()
        bucket = int(hash_val[:8], 16) / 0xFFFFFFFF
        
        cumulative = 0
        for variant_name, ratio in experiment.allocation.items():
            cumulative += ratio
            if bucket < cumulative:
                return variant_name
        return list(experiment.allocation.keys())[0]
```

## 7. 持续评估

### 7.1 持续评估架构

```
┌─────────────────────────────────────────────────────┐
│              持续评估流水线                            │
│                                                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ 生产流量  │ →  │ 采样器   │ →  │ 评估队列  │       │
│  │ (100%)   │    │ (10%)    │    │ (异步)    │       │
│  └──────────┘    └──────────┘    └──────────┘       │
│                                        │             │
│                                 ┌──────▼──────┐      │
│                                 │  评估引擎   │      │
│                                 │  (多指标)   │      │
│                                 └──────┬──────┘      │
│                                        │             │
│                        ┌───────────────┼──────────┐  │
│                        ▼               ▼          ▼  │
│                 ┌──────────┐   ┌──────────┐ ┌─────┐ │
│                 │ Dashboard│   │  告警    │ │回滚 │ │
│                 │ (趋势)   │   │ (阈值)   │ │触发 │ │
│                 └──────────┘   └──────────┘ └─────┘ │
└─────────────────────────────────────────────────────┘
```

### 7.2 退化检测

```python
class DegradationDetector:
    """质量退化检测器"""
    
    def __init__(self, window_size: int = 100,
                 sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity  # 标准差倍数
        self.history = deque(maxlen=1000)
    
    def add_score(self, score: float):
        self.history.append(score)
    
    def is_degrading(self) -> tuple[bool, dict]:
        """检测是否发生退化"""
        if len(self.history) < self.window_size * 2:
            return False, {"reason": "数据不足"}
        
        recent = list(self.history)[-self.window_size:]
        baseline = list(self.history)[-self.window_size*2:-self.window_size]
        
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)
        recent_mean = np.mean(recent)
        
        z_score = (baseline_mean - recent_mean) / (baseline_std + 1e-6)
        
        is_degrading = z_score > self.sensitivity
        
        return is_degrading, {
            "baseline_mean": baseline_mean,
            "recent_mean": recent_mean,
            "z_score": z_score,
            "drop_percentage": (baseline_mean - recent_mean) / baseline_mean * 100
        }
```

## 8. 回滚策略

### 8.1 回滚类型

```
即时回滚（秒级）：
  • 切换 Prompt 版本（配置中心热更新）
  • 切换流量路由（负载均衡配置）
  适用：Prompt 变更、配置变更

快速回滚（分钟级）：
  • 切换模型版本（预加载的备用模型）
  • 切换 RAG 索引（蓝绿索引切换）
  适用：模型变更、索引变更

全量回滚（小时级）：
  • 重建索引 + 重新部署
  适用：严重数据问题
```

### 8.2 回滚决策

```python
class RollbackManager:
    """回滚管理器"""
    
    # 自动回滚条件
    auto_rollback_rules = {
        "critical": {
            "error_rate": {"threshold": 0.05, "window": "5m"},
            "hallucination_rate": {"threshold": 0.10, "window": "10m"},
        },
        "warning": {
            "quality_drop": {"threshold": 0.15, "window": "30m"},
            "latency_spike": {"threshold": 2.0, "window": "15m"},  # 2x baseline
        }
    }
    
    async def check_and_rollback(self):
        for rule_name, condition in self.auto_rollback_rules["critical"].items():
            metric_value = await self.metrics.get(rule_name, condition["window"])
            if metric_value > condition["threshold"]:
                await self._execute_rollback(
                    reason=f"Critical: {rule_name}={metric_value}"
                )
                return
    
    async def _execute_rollback(self, reason: str):
        """执行回滚"""
        logger.critical(f"Initiating rollback: {reason}")
        
        # 1. 切换流量到上一个稳定版本
        previous_version = self.registry.get_previous_production()
        await self.router.switch_to(previous_version)
        
        # 2. 通知团队
        await self.alerter.send(
            severity="critical",
            message=f"Auto-rollback triggered: {reason}"
        )
        
        # 3. 记录事件
        self.registry.record_rollback(reason=reason)
```

## 9. 完整 CI/CD 流水线

```
┌─────────────────────────────────────────────────────────┐
│           LLMOps CI/CD Pipeline                          │
│                                                           │
│  Commit → Build → Test → Evaluate → Stage → Deploy      │
│                                                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │  Build  │  │  Test   │  │Evaluate │                 │
│  │         │  │         │  │         │                 │
│  │• Docker │  │• Unit   │  │• Ragas  │                 │
│  │• Deps   │  │• Integ  │  │• Custom │                 │
│  │• Config │  │• E2E    │  │• Compare│                 │
│  └────┬────┘  └────┬────┘  └────┬────┘                 │
│       └─────────────┼─────────────┘                      │
│                     ▼                                    │
│              ┌──────────────┐                            │
│              │ Quality Gate │  ← 所有指标必须达标        │
│              └──────┬───────┘                            │
│                     │ Pass                               │
│              ┌──────▼───────┐                            │
│              │ Shadow Test  │  ← 影子测试 24h           │
│              └──────┬───────┘                            │
│                     │ OK                                 │
│              ┌──────▼───────┐                            │
│              │ Canary 1%   │  ← 灰度开始               │
│              └──────┬───────┘                            │
│                     │ ...                                │
│              ┌──────▼───────┐                            │
│              │ Full Deploy  │  ← 全量发布               │
│              └──────────────┘                            │
└─────────────────────────────────────────────────────────┘
```

## 10. 小结

从实验到部署的关键原则：

1. **一切可追溯**：每次变更都有实验记录和评估结果
2. **评估门禁**：未通过评估的变更不得进入生产
3. **渐进式发布**：影子测试 → 灰度 → 全量
4. **持续评估**：上线后继续监控质量
5. **快速回滚**：发现问题秒级/分钟级回滚
6. **自动化**：尽量自动化决策，人工只处理边界情况

对于 8 × H20 的环境，建议：
- 用 MLflow 管理实验和模型注册
- 用 Langfuse 做全链路追踪
- 用 Kubernetes + Istio 实现灰度流量管理
- 用 Prometheus + Grafana 做监控告警
