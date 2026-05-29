# 05 - GPU 容量规划方法论

## 容量规划的核心问题

```
"我需要多少张 GPU 才能满足业务需求？"

这个问题的答案取决于：
1. SLO 要求（延迟、吞吐、可用性）
2. 流量模型（QPS、分布、增长率）
3. 模型特征（大小、量化、序列长度）
4. 硬件能力（GPU 型号、互联拓扑）
5. 冗余需求（N+1、跨 AZ）
```

---

## 从 SLO 反推 GPU 需求

### 基本公式

```
需要的 GPU 数量 = f(目标吞吐量, 单 GPU 吞吐量, 冗余系数)

展开：
  target_throughput = peak_qps × avg_output_tokens × headroom_factor
  single_gpu_throughput = f(model_size, quantization, batch_size, slo_constraints)
  gpu_count = ceil(target_throughput / single_gpu_throughput) × tp_degree × redundancy

其中：
  peak_qps: 峰值请求率
  avg_output_tokens: 平均输出 token 数
  headroom_factor: 余量系数（通常 1.3-1.5）
  tp_degree: Tensor Parallelism 度（H20 通常 TP=8 for 70B）
  redundancy: 冗余系数（N+1 或 N+2）
```

### 详细计算模型

```python
class CapacityPlanner:
    """GPU 容量规划计算器"""

    def __init__(self, gpu_config, model_config, slo_config):
        self.gpu = gpu_config
        self.model = model_config
        self.slo = slo_config

    def calculate_single_instance_capacity(self):
        """计算单实例（可能多 GPU TP）的吞吐能力"""

        # Prefill 吞吐（compute-bound）
        # H20 FP8: 148 TFLOPS per GPU
        flops_per_token = 2 * self.model.params  # 大约估算
        prefill_tokens_per_second = (
            self.gpu.flops * self.gpu.count_per_instance
            * self.gpu.compute_efficiency  # 通常 0.4-0.6
            / flops_per_token
        )

        # Decode 吞吐（memory-bound）
        # H20 HBM: 4TB/s per GPU
        bytes_per_token = self.model.kv_cache_bytes_per_token
        model_bytes = self.model.size_bytes
        decode_tokens_per_second = (
            self.gpu.memory_bandwidth * self.gpu.count_per_instance
            * self.gpu.bandwidth_efficiency  # 通常 0.6-0.8
            / (model_bytes + bytes_per_token * self.model.avg_context_length)
        )

        # 实际吞吐受 SLO 约束
        # 当吞吐增加时，延迟也会增加（批处理效应）
        max_throughput = min(prefill_tokens_per_second, decode_tokens_per_second)

        # SLO 约束下的有效吞吐（通常是最大吞吐的 60-80%）
        slo_constrained_throughput = max_throughput * self.slo.throughput_factor

        return slo_constrained_throughput

    def calculate_total_gpus(self, traffic_model):
        """计算总 GPU 需求"""

        single_instance_tps = self.calculate_single_instance_capacity()

        # 峰值需求
        peak_tokens_per_second = (
            traffic_model.peak_qps
            * traffic_model.avg_output_tokens
        )

        # 加上 headroom
        required_tokens_per_second = peak_tokens_per_second * traffic_model.headroom

        # 需要的实例数
        instances_needed = math.ceil(required_tokens_per_second / single_instance_tps)

        # 冗余
        instances_with_redundancy = math.ceil(
            instances_needed * traffic_model.redundancy_factor
        )

        # GPU 总数
        total_gpus = instances_with_redundancy * self.gpu.count_per_instance

        return {
            "single_instance_tps": single_instance_tps,
            "peak_demand_tps": peak_tokens_per_second,
            "instances_needed": instances_needed,
            "instances_with_redundancy": instances_with_redundancy,
            "total_gpus": total_gpus,
        }
```

### 计算示例

```python
# H20 + Qwen2-72B 场景
gpu_config = GPUConfig(
    model="H20",
    flops=148e12,        # 148 TFLOPS FP8
    memory_bandwidth=4e12, # 4 TB/s
    memory_size=96e9,      # 96 GB
    count_per_instance=8,  # TP=8
    compute_efficiency=0.5,
    bandwidth_efficiency=0.7,
)

model_config = ModelConfig(
    name="qwen2-72b",
    params=72e9,
    size_bytes=36e9,       # INT4 量化后约 36GB
    kv_cache_bytes_per_token=0.5e6,  # 每 token 0.5MB KV Cache
    avg_context_length=4000,
)

slo_config = SLOConfig(
    ttft_p99_ms=800,
    tpot_p99_ms=80,
    throughput_factor=0.7,  # SLO 约束下只能用 70% 的理论吞吐
)

traffic_model = TrafficModel(
    peak_qps=100,              # 峰值 100 QPS
    avg_output_tokens=300,     # 平均生成 300 tokens
    headroom=1.3,              # 30% 余量
    redundancy_factor=1.25,    # N+1 冗余（4 实例加 1 备份）
)

# 计算结果（大致估算）：
# single_instance_tps ≈ 2000 tokens/s
# peak_demand_tps = 100 * 300 = 30,000 tokens/s
# with headroom = 39,000 tokens/s
# instances_needed = ceil(39000/2000) = 20
# with redundancy = ceil(20 * 1.25) = 25
# total_gpus = 25 * 8 = 200 张 H20
```

---

## 流量预测模型

### 流量特征分析

```
AI 推理流量的典型特征：
1. 日内波动（工作时间 vs 夜间）
2. 周期性（工作日 vs 周末）
3. 突发性（新功能上线、营销活动）
4. 长尾分布（大部分请求短，少数请求很长）
5. 增长趋势（用户增长、新场景接入）
```

### 时间序列预测方法

```python
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class TrafficForecaster:
    """流量预测器"""

    def __init__(self, historical_data: np.ndarray, period: int = 24):
        """
        historical_data: 历史 QPS 数据（按小时）
        period: 周期长度（24 = 日周期）
        """
        self.data = historical_data
        self.period = period

    def forecast_holt_winters(self, hours_ahead: int = 168):
        """Holt-Winters 三重指数平滑（处理趋势 + 季节性）"""
        model = ExponentialSmoothing(
            self.data,
            trend='add',
            seasonal='mul',
            seasonal_periods=self.period,
        ).fit()

        forecast = model.forecast(hours_ahead)
        return forecast

    def forecast_with_confidence(self, hours_ahead: int = 168,
                                  confidence: float = 0.95):
        """带置信区间的预测"""
        forecast = self.forecast_holt_winters(hours_ahead)

        # 使用历史残差估计预测区间
        residuals = self.data - self.model.fittedvalues
        std = np.std(residuals)

        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        upper = forecast + z * std * np.sqrt(np.arange(1, hours_ahead + 1))
        lower = forecast - z * std * np.sqrt(np.arange(1, hours_ahead + 1))

        return {
            "forecast": forecast,
            "upper_bound": upper,
            "lower_bound": lower,
        }

    def calculate_peak_multiplier(self):
        """计算峰值倍数（Peak / Average）"""
        hourly_avg = np.mean(self.data.reshape(-1, 24), axis=0)
        peak = np.max(hourly_avg)
        mean = np.mean(hourly_avg)
        return peak / mean
```

### 容量规划时间线

```
短期（1-7 天）：
  方法：基于历史模式 + 已知事件
  精度：±10%
  用途：日常调度，Spot 实例预留

中期（1-3 个月）：
  方法：Holt-Winters + 业务增长曲线
  精度：±20%
  用途：Reserved Instance 采购，预算规划

长期（6-12 个月）：
  方法：业务预测 + GPU 价格趋势
  精度：±40%
  用途：硬件采购，数据中心规划
```

---

## 弹性扩缩容策略

### 推理服务扩缩容的特殊挑战

```
挑战 1: 冷启动时间长
  传统微服务：Pod 启动 2-10 秒
  推理服务：模型加载 30 秒 - 5 分钟
  → 不能等到流量来了再扩容

挑战 2: GPU 资源碎片化
  需要 8 张连续的 GPU（TP=8）
  集群有 20 张空闲 GPU 但分散在不同节点
  → 无法扩容

挑战 3: KV Cache 状态
  扩容的新实例没有 Prefix Cache
  同模型的请求被路由到新实例，TTFT 上升
  → 需要 Cache 预热策略

挑战 4: 缩容风险
  缩容时要等进行中的请求完成
  长请求可能需要几分钟
  → 需要 graceful shutdown
```

### 多层扩缩容策略

```yaml
scaling_policy:
  # Layer 1: 预测性扩缩容（基于流量预测）
  predictive:
    enabled: true
    forecast_window: 2h       # 提前 2 小时预测
    scale_up_lead_time: 10m   # 扩容需要 10 分钟（模型加载）
    confidence_level: 0.95
    action: scale_to_predicted_peak

  # Layer 2: 响应式扩缩容（基于实时指标）
  reactive:
    scale_up:
      metrics:
        - name: queue_depth
          threshold: 50
          duration: 60s
        - name: ttft_p99
          threshold_ratio: 0.8  # SLO 的 80%
          duration: 120s
        - name: gpu_utilization_sm_active
          threshold: 0.85
          duration: 300s
      cooldown: 300s
      max_step: 2  # 每次最多扩 2 个实例

    scale_down:
      metrics:
        - name: gpu_utilization_sm_active
          threshold: 0.3
          duration: 900s  # 15 分钟低于阈值才缩容
        - name: queue_depth
          threshold: 5
          duration: 600s
      cooldown: 600s
      max_step: 1  # 每次最多缩 1 个实例
      graceful_period: 300s  # 等待进行中的请求完成

  # Layer 3: 紧急扩容（突发流量）
  emergency:
    trigger: queue_depth > 200 AND ttft_p99 > slo_target
    action: scale_up_immediately
    max_instances: cluster_max  # 扩到集群上限
    notification: pager  # 通知值班人员
```

---

## Spot 实例对容量的影响

### Spot 实例风险分析

```
Spot 实例优势：
  - 成本节省 60-90%
  - 适合可中断的批处理任务

Spot 实例风险（推理场景）：
  - 2 分钟中断通知（可能不够完成长请求）
  - 可用性不保证（高峰期可能拿不到）
  - 中断导致请求失败，影响 SLO

适用场景分析：
  ┌─────────────────┬───────────┬────────────────┐
  │ 场景            │ Spot 适用性│ 原因            │
  ├─────────────────┼───────────┼────────────────┤
  │ 在线推理 (SLO)  │ ✗ 不推荐   │ 中断影响用户体验│
  │ 在线推理 (溢出) │ △ 谨慎     │ 作为额外容量    │
  │ 异步 API        │ ○ 可以     │ 可重试          │
  │ 批量处理        │ ★ 推荐     │ 可中断可恢复    │
  │ 开发测试        │ ★ 推荐     │ 无 SLO 要求     │
  │ 模型评测        │ ★ 推荐     │ 可重跑          │
  └─────────────────┴───────────┴────────────────┘
```

### 混合容量策略

```
总容量 = Base Capacity + Reserved Buffer + Spot Buffer

Base Capacity（基础容量）：
  - On-Demand 或 Reserved Instances
  - 覆盖日常负载的 70-80%
  - SLO 保证

Reserved Buffer（预留缓冲）：
  - Reserved Instances
  - 覆盖可预测的峰值
  - 长期合约更划算

Spot Buffer（弹性缓冲）：
  - Spot Instances
  - 覆盖突发流量
  - 需要中断处理机制
```

```
                     ┌───────────────────── Spot Buffer (突发)
                     │
                     │  ┌────────────────── Reserved Buffer (峰值)
                     │  │
  QPS ──────────────────────────────
                     │  │  │
                     │  │  │  ┌─────── Base Capacity (基础)
                     │  │  │  │
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                     │  │  │  │
                 突发峰值 日峰值 均值
```

---

## 容量规划自动化

### 持续容量评估

```python
class ContinuousCapacityEvaluator:
    """持续评估当前容量是否足够"""

    def evaluate(self):
        current_metrics = self.get_current_metrics()
        forecast = self.forecaster.forecast_with_confidence(hours_ahead=168)

        report = {
            "current_state": self.assess_current(current_metrics),
            "7_day_forecast": self.assess_forecast(forecast),
            "recommendations": [],
        }

        # 检查是否需要扩容
        peak_demand = forecast["upper_bound"].max()
        current_capacity = current_metrics["max_throughput"]

        headroom = (current_capacity - peak_demand) / current_capacity

        if headroom < 0.1:
            report["recommendations"].append({
                "urgency": "critical",
                "action": "scale_up",
                "reason": f"预测峰值需求将超出当前容量 {headroom:.0%} 余量",
                "suggested_gpus": self.calculate_additional_gpus(peak_demand),
            })
        elif headroom < 0.3:
            report["recommendations"].append({
                "urgency": "warning",
                "action": "plan_scale_up",
                "reason": f"容量余量仅 {headroom:.0%}，建议提前规划扩容",
            })
        elif headroom > 0.6:
            report["recommendations"].append({
                "urgency": "info",
                "action": "consider_scale_down",
                "reason": f"容量余量 {headroom:.0%}，可考虑缩容节省成本",
            })

        return report
```

### 容量规划决策树

```
                    ┌─ SLO 达标？
                    │
            ┌───────┴──────┐
            │ Yes          │ No
            │              │
     ┌──────┴─────┐   ┌───┴────────────┐
     │余量 > 30%? │   │ 哪个 SLI 违规？ │
     │            │   │                │
  ┌──┴──┐     ┌──┴──┐ │               │
  │Yes  │     │No   │ ├── TTFT → 扩容或优化 Prefill
  │     │     │     │ ├── TPOT → 扩容或优化 Decode
  │考虑 │     │维持 │ ├── Queue → 扩容
  │缩容 │     │现状 │ └── Error → 排查根因
  └─────┘     └─────┘
```

---

## 容量规划与成本的权衡

```
                成本
                 ↑
                 │            ╱ Over-provisioned
                 │           ╱  (过度配置)
                 │          ╱
                 │         ╱
    Sweet Spot → │ ── ── ●── ── ── ── 最优点
                 │       ╱│
                 │      ╱ │
                 │     ╱  │
                 │    ╱   │ Under-provisioned
                 │   ╱    │ (SLO 违规)
                 │──╱─────┼──────────→ 容量
                 │        │
              SLO 阈值    │
                          │
              SLO 违规开始的临界点

目标：找到 Sweet Spot —— 刚好满足 SLO，不过度浪费
方法：持续监控 + 自动化调整 + 定期复盘
```

---

## 下一步

→ 进入 [06_finops_for_ai.md](06_finops_for_ai.md) 了解 GPU 成本的全面管理
