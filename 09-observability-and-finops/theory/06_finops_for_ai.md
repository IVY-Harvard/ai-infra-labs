# 06 - AI FinOps：GPU 成本的全面管理

## GPU 成本的真实构成

### 远不止"卡时"这么简单

```
一张 H20 的 TCO（Total Cost of Ownership）分解：

┌────────────────────────────────────────────────────────┐
│ 组件              │ 云上（按需）  │ 自建 IDC       │
├───────────────────┼──────────────┼───────────────────┤
│ GPU 硬件/租赁     │ 60-70%       │ 35-40%（折旧）  │
│ 电力 + 散热       │ 含在内       │ 20-30%          │
│ 网络带宽          │ 5-10%        │ 5-8%            │
│ 存储（模型+日志） │ 5-15%        │ 8-12%           │
│ 运维人力          │ 3-5%         │ 15-20%          │
│ 软件许可          │ 2-5%         │ 3-5%            │
│ 其他（安全/合规） │ 3-5%         │ 5-8%            │
└───────────────────┴──────────────┴───────────────────┘

关键洞察：
  - 云上成本主要是 GPU 计算费
  - 自建 IDC 的电力和运维隐性成本很高
  - 8 张 H20 的年 TCO（云上按需）约 $800K-$1.2M
  - 如果利用率只有 40%，每年浪费 $400K-$600K
```

### GPU 小时的精确计算

```python
class GPUCostModel:
    """GPU 成本精确计算模型"""

    def __init__(self, pricing_config):
        self.pricing = pricing_config

    def calculate_gpu_hour_cost(self, gpu_type: str, pricing_tier: str):
        """计算单 GPU 小时的全成本"""

        base_price = self.pricing.get_base_price(gpu_type, pricing_tier)

        # 附加成本
        network_cost = base_price * 0.08      # 网络约 8%
        storage_cost = base_price * 0.05      # 存储约 5%
        overhead_cost = base_price * 0.03     # 管理开销 3%

        total_cost_per_gpu_hour = base_price + network_cost + storage_cost + overhead_cost

        return {
            "base": base_price,
            "network": network_cost,
            "storage": storage_cost,
            "overhead": overhead_cost,
            "total": total_cost_per_gpu_hour,
        }

    def calculate_inference_cost(self, request_metrics):
        """计算单次推理请求的成本"""

        gpu_seconds_used = request_metrics["gpu_time_seconds"]
        gpu_count = request_metrics["gpu_count"]  # TP 度

        cost_per_second = self.gpu_hour_cost / 3600
        request_cost = cost_per_second * gpu_seconds_used * gpu_count

        # 加上排队等待期间的 GPU 空闲成本（重要！）
        queue_wait_seconds = request_metrics.get("queue_wait_seconds", 0)
        # 排队时 GPU 并非完全空闲（在处理其他请求），但有部分资源被预留
        idle_fraction = request_metrics.get("idle_fraction_during_queue", 0.1)
        queue_cost = cost_per_second * queue_wait_seconds * gpu_count * idle_fraction

        return {
            "compute_cost": request_cost,
            "queue_cost": queue_cost,
            "total_cost": request_cost + queue_cost,
            "cost_per_1k_tokens": (request_cost + queue_cost) /
                                   (request_metrics["total_tokens"] / 1000),
        }
```

---

## 成本分摊模型

### 为什么需要成本分摊

```
场景：你的 8 张 H20 服务了 3 个业务方：
  - Team A：在线客服 Bot，低延迟高优先
  - Team B：文档摘要服务，中延迟中优先
  - Team C：离线数据标注，无 SLO

问题：这个月 GPU 账单怎么分？

错误方法：
  ✗ 平均分 → 对用量少的不公平
  ✗ 按请求数分 → 长请求和短请求差异巨大
  ✗ 按 GPU 时间分 → 忽略了优先级溢价

正确方法：多维度加权分摊
```

### 分摊维度设计

```python
class ChargebackModel:
    """多维度成本分摊模型"""

    # 分摊维度及权重
    DIMENSIONS = {
        "gpu_time": 0.50,        # GPU 实际计算时间（主要维度）
        "gpu_memory": 0.20,      # 显存占用（KV Cache 等）
        "priority_premium": 0.15, # 优先级溢价
        "reservation": 0.15,      # 资源预留（即使没用也要付费）
    }

    def calculate_chargeback(self, tenant_id: str, period: str):
        """计算某租户在某周期的费用"""

        metrics = self.get_tenant_metrics(tenant_id, period)
        total_cluster_cost = self.get_total_cost(period)

        # 维度 1: GPU 时间占比
        gpu_time_share = metrics["gpu_seconds"] / self.total_gpu_seconds(period)

        # 维度 2: 显存占用占比（加权平均）
        memory_share = metrics["avg_memory_gb"] / self.total_memory_capacity()

        # 维度 3: 优先级溢价
        priority_multiplier = {
            "critical": 2.0,    # 最高优先，保证资源
            "high": 1.5,        # 高优先
            "normal": 1.0,      # 普通
            "low": 0.6,         # 低优先，可被抢占
            "batch": 0.3,       # 批处理，用剩余资源
        }[metrics["priority"]]

        # 维度 4: 资源预留
        reserved_share = metrics["reserved_gpus"] / self.total_gpus()

        # 加权计算
        weighted_share = (
            self.DIMENSIONS["gpu_time"] * gpu_time_share +
            self.DIMENSIONS["gpu_memory"] * memory_share +
            self.DIMENSIONS["priority_premium"] * (gpu_time_share * priority_multiplier) +
            self.DIMENSIONS["reservation"] * reserved_share
        )

        # 归一化（所有租户的 weighted_share 之和应为 1）
        normalized_share = weighted_share / self.total_weighted_shares(period)

        return {
            "tenant_id": tenant_id,
            "period": period,
            "total_cost": total_cluster_cost * normalized_share,
            "breakdown": {
                "compute": total_cluster_cost * self.DIMENSIONS["gpu_time"] * gpu_time_share,
                "memory": total_cluster_cost * self.DIMENSIONS["gpu_memory"] * memory_share,
                "priority": total_cluster_cost * self.DIMENSIONS["priority_premium"] * gpu_time_share * priority_multiplier,
                "reservation": total_cluster_cost * self.DIMENSIONS["reservation"] * reserved_share,
            },
        }
```

### 按项目/模型的成本归因

```
成本归因矩阵：

┌──────────────┬───────────┬───────────┬───────────┬──────────┐
│ 模型          │ Team A    │ Team B    │ Team C    │ 内部研发  │
├──────────────┼───────────┼───────────┼───────────┼──────────┤
│ Qwen2-72B    │ $12,400   │ $8,200    │ $3,100    │ $1,800   │
│ Qwen2-7B     │ $2,100    │ $5,600    │ $7,200    │ $900     │
│ Embedding    │ $800      │ $1,200    │ $4,500    │ $200     │
├──────────────┼───────────┼───────────┼───────────┼──────────┤
│ 小计          │ $15,300   │ $15,000   │ $14,800   │ $2,900   │
│ 优先级溢价    │ +$4,590   │ +$2,250   │ -$2,960   │ $0       │
├──────────────┼───────────┼───────────┼───────────┼──────────┤
│ 最终账单      │ $19,890   │ $17,250   │ $11,840   │ $2,900   │
└──────────────┴───────────┴───────────┴───────────┴──────────┘

关键：高优先级租户付出溢价，低优先级租户享受折扣
这是公平的——高优先意味着占用了更多的"保障能力"
```

---

## Spot vs Reserved vs On-Demand 策略

### 三种定价模式对比

```
┌──────────────┬────────────┬────────────┬────────────────────────┐
│ 类型         │ 折扣       │ 承诺       │ 适用场景               │
├──────────────┼────────────┼────────────┼────────────────────────┤
│ On-Demand    │ 0%         │ 无         │ 短期突发、POC          │
│ Reserved 1Y  │ 30-40%     │ 1 年       │ 稳定基础负载           │
│ Reserved 3Y  │ 50-60%     │ 3 年       │ 长期确定性需求         │
│ Spot         │ 60-90%     │ 无（可中断）│ 批处理、训练、评测     │
│ Savings Plan │ 20-35%     │ 消费承诺   │ 跨实例类型的灵活承诺   │
└──────────────┴────────────┴────────────┴────────────────────────┘
```

### 最优组合策略

```python
class PricingOptimizer:
    """GPU 定价组合优化器"""

    def optimize_fleet_composition(self, demand_profile):
        """
        输入：需求曲线（每小时需要多少 GPU）
        输出：最优的 Reserved/On-Demand/Spot 配比
        """

        # 分析需求分布
        demand_array = np.array(demand_profile)
        p10 = np.percentile(demand_array, 10)   # 最低需求
        p50 = np.percentile(demand_array, 50)   # 中位需求
        p90 = np.percentile(demand_array, 90)   # 峰值需求
        p99 = np.percentile(demand_array, 99)   # 极端峰值

        # 策略：层次化覆盖
        reserved_3y = int(p10 * 0.9)   # 最低需求的 90% 用 3 年 RI
        reserved_1y = int((p50 - reserved_3y) * 0.8)  # 中位到基线的差用 1 年 RI
        on_demand_base = int(p90 - reserved_3y - reserved_1y)  # 峰值用按需
        spot_buffer = int((p99 - p90) * 1.5)  # 极端峰值用 Spot（多买点防中断）

        # 成本计算
        monthly_cost = (
            reserved_3y * self.price_reserved_3y * 730 +  # 730 小时/月
            reserved_1y * self.price_reserved_1y * 730 +
            on_demand_base * self.price_on_demand * self.avg_on_demand_hours +
            spot_buffer * self.price_spot * self.avg_spot_hours
        )

        # 对比纯按需的节省
        pure_on_demand_cost = p90 * self.price_on_demand * 730
        savings = 1 - (monthly_cost / pure_on_demand_cost)

        return {
            "reserved_3y_gpus": reserved_3y,
            "reserved_1y_gpus": reserved_1y,
            "on_demand_gpus": on_demand_base,
            "spot_gpus": spot_buffer,
            "monthly_cost": monthly_cost,
            "savings_vs_on_demand": f"{savings:.1%}",
        }
```

### 8 张 H20 的实际策略建议

```
你的情况：8 张 H20，混合用于推理和开发

推荐策略：
  ┌─────────────────────────────────────────────────┐
  │ GPU 1-4: Reserved Instance (1Y)                 │
  │   用途：在线推理服务（有 SLO）                    │
  │   理由：稳定负载，1 年 RI 节省 35%               │
  │                                                 │
  │ GPU 5-6: On-Demand                              │
  │   用途：弹性推理 + 开发测试                      │
  │   理由：需求波动大，按需灵活                     │
  │                                                 │
  │ GPU 7-8: 混合策略                               │
  │   白天：On-Demand 补充推理容量                   │
  │   夜间：跑训练/评测任务（可以用 Spot 补充）       │
  │   理由：最大化利用率                             │
  └─────────────────────────────────────────────────┘

  预期节省：相比全部 On-Demand 节省约 25-30%
```

---

## 闲置检测与回收

### GPU 闲置的多层定义

```
Level 0: 完全空闲
  - 无任何进程在 GPU 上
  - 检测方法：nvidia-smi 无进程
  - 回收策略：立即释放

Level 1: 分配但空闲
  - 有进程持有 GPU（如 vLLM 已启动）
  - 但无实际推理请求
  - 检测方法：SM Active = 0% 超过阈值
  - 回收策略：缩容或接受其他工作负载

Level 2: 严重低利用
  - 有少量请求，但利用率极低
  - SM Active < 10% 持续 30 分钟
  - 检测方法：DCGM 指标持续低于阈值
  - 回收策略：合并工作负载，释放部分 GPU

Level 3: 低效利用
  - 利用率不低，但效率差
  - 高 SM Active 但低 Tensor Core Util
  - 检测方法：效率指标异常
  - 回收策略：优化而非回收（Profile 分析）
```

### 闲置检测引擎

```python
class IdleDetector:
    """GPU 闲置检测引擎"""

    IDLE_THRESHOLDS = {
        "level_0": {
            "condition": "no_process",
            "duration_minutes": 5,
            "action": "immediate_release",
        },
        "level_1": {
            "condition": "sm_active < 1%",
            "duration_minutes": 15,
            "action": "notify_owner_then_release",
        },
        "level_2": {
            "condition": "sm_active < 10% AND memory_util < 20%",
            "duration_minutes": 30,
            "action": "consolidate_workloads",
        },
        "level_3": {
            "condition": "tensor_core_active < 5% AND sm_active > 50%",
            "duration_minutes": 60,
            "action": "recommend_optimization",
        },
    }

    def scan_idle_gpus(self) -> list:
        """扫描所有 GPU 的闲置状态"""
        results = []

        for gpu_id in range(self.total_gpus):
            metrics = self.get_gpu_metrics(gpu_id)
            idle_level = self.classify_idle_level(metrics)

            if idle_level is not None:
                idle_duration = self.get_idle_duration(gpu_id, idle_level)
                threshold = self.IDLE_THRESHOLDS[idle_level]

                if idle_duration >= threshold["duration_minutes"] * 60:
                    results.append({
                        "gpu_id": gpu_id,
                        "level": idle_level,
                        "duration_seconds": idle_duration,
                        "owner": self.get_gpu_owner(gpu_id),
                        "suggested_action": threshold["action"],
                        "potential_savings_per_hour": self.calculate_savings(gpu_id),
                    })

        return sorted(results, key=lambda x: x["potential_savings_per_hour"], reverse=True)

    def auto_reclaim(self, idle_gpu: dict):
        """自动回收闲置 GPU"""
        action = idle_gpu["suggested_action"]

        if action == "immediate_release":
            self.release_gpu(idle_gpu["gpu_id"])
        elif action == "notify_owner_then_release":
            self.notify_owner(idle_gpu["owner"], idle_gpu)
            # 通知后等待 grace period
            self.schedule_release(idle_gpu["gpu_id"], delay_minutes=10)
        elif action == "consolidate_workloads":
            self.trigger_consolidation(idle_gpu["gpu_id"])
```

### 资源回收策略

```yaml
# 闲置回收策略配置
idle_reclamation:
  enabled: true

  # 扫描频率
  scan_interval: 5m

  # 分级策略
  policies:
    - level: development
      idle_threshold: 30m
      action: suspend_and_notify
      grace_period: 10m
      working_hours_only: true  # 只在工作时间回收

    - level: staging
      idle_threshold: 1h
      action: scale_to_zero
      grace_period: 15m

    - level: production
      idle_threshold: never  # 生产环境不自动回收
      action: alert_only

  # 例外规则
  exceptions:
    - label: "keep-alive=true"  # 有此标签的不回收
    - namespace: "model-serving"  # 生产推理服务不回收
    - time_range: "02:00-06:00"  # 凌晨可能在跑批任务

  # 回收后的资源去向
  reclaimed_pool:
    priority_order:
      - pending_jobs_high_priority
      - pending_jobs_normal
      - spot_pool  # 放入 Spot 池供低优先级任务
```

---

## 成本可视化与报告

### 关键成本 KPI

```
必须追踪的成本 KPI：

1. Cost per 1K tokens（每千 token 成本）
   公式：总 GPU 成本 / 总处理 token 数 × 1000
   基准（H20 + 72B FP8）：~$0.003-0.008 / 1K tokens
   对标：GPT-4o API $0.005 / 1K output tokens

2. GPU 利用率加权成本
   公式：GPU 成本 / (GPU 利用率 × 有效计算比例)
   目标：有效利用率 > 60%

3. 单请求 P50/P99 成本
   用于异常检测——突然变贵的请求可能有问题

4. 成本效率比（Revenue per GPU-hour）
   公式：该 GPU 产生的业务价值 / GPU 成本
   目标：> 3x（即每花 1 元 GPU 产生 3 元业务价值）

5. 闲置浪费率
   公式：闲置 GPU 时间 / 总 GPU 时间
   目标：< 15%
```

### 成本异常检测

```python
class CostAnomalyDetector:
    """成本异常检测"""

    def detect_anomalies(self, current_period_cost: dict, historical_costs: list):
        """
        检测成本是否异常偏离历史基线

        常见异常模式：
        1. 突增：某租户成本突然翻倍（可能有 bug 导致重试风暴）
        2. 阶梯跳变：持续高于基线（可能模型配置变更）
        3. 周期异常：打破了正常的日/周周期（需排查）
        """
        anomalies = []

        for tenant, cost in current_period_cost.items():
            historical = [h[tenant] for h in historical_costs if tenant in h]

            if len(historical) < 7:
                continue

            mean = np.mean(historical)
            std = np.std(historical)

            # Z-score 检测
            z_score = (cost - mean) / std if std > 0 else 0

            if abs(z_score) > 3:
                anomalies.append({
                    "tenant": tenant,
                    "current_cost": cost,
                    "expected_cost": mean,
                    "deviation": f"{z_score:.1f} sigma",
                    "severity": "critical" if z_score > 5 else "warning",
                    "possible_causes": self.analyze_root_cause(tenant),
                })

        return anomalies
```

---

## 成本优化实践清单

```
立即可做的优化（Quick Wins）：

□ 启用闲置检测，回收开发环境空闲 GPU
□ 夜间低峰期跑批处理任务（填满 GPU）
□ 对低优先级请求启用更大的 batch size（提高吞吐，降低单请求成本）
□ 检查是否有忘记释放的 GPU（开发者跑完实验没清理）

中期优化（1-2 周）：

□ 部署多模型共享 GPU（小模型共享，避免碎片化）
□ 实施请求路由优化（相似前缀的请求路由到同实例，提高 Prefix Cache 命中）
□ 建立成本分摊仪表盘，让每个团队看到自己的消耗
□ 评估 Reserved Instance 采购（稳定负载用 RI 节省 30-40%）

长期优化（1-3 个月）：

□ 实施模型蒸馏（72B → 7B 对简单任务够用，成本降 90%）
□ 建立请求分级路由（简单问题用小模型，复杂问题用大模型）
□ 部署 KV Cache 共享层（跨实例共享热点前缀）
□ 评估自建 IDC vs 云的长期成本差异
```

---

## 下一步

→ 进入 [07_aiops_automation.md](07_aiops_automation.md) 学习如何用 AI 自动化 AI Infra 运维
