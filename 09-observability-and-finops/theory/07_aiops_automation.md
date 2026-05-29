# 07 - AIOps：用 AI 做 AI Infra 运维

## 为什么 AI Infra 需要 AIOps

```
传统监控的困境：

你的 8 张 H20 每秒产生的指标数据：
  - DCGM 指标：~200 个/GPU × 8 GPU = 1,600 个时间序列
  - vLLM 指标：~50 个/实例 × N 实例 = 数百个时间序列
  - 系统指标：CPU/Memory/Network/Disk = 数十个
  - 应用日志：~1000 条/分钟
  - 追踪数据：每个请求 5-10 个 span

人工不可能实时关注所有这些指标。
传统阈值告警要么漏报（阈值太高）要么误报（阈值太低）。

AIOps 的核心价值：
  1. 从海量指标中自动发现异常模式
  2. 关联多维指标，自动进行根因分析
  3. 基于历史 pattern 进行预测性维护
  4. 自动执行标准化修复操作
  5. 通过 ChatOps 降低运维门槛
```

---

## 异常检测方法论

### 方法一：统计方法

```python
"""
统计方法适用于：
- 指标有明显的正态分布特征
- 需要快速实现、低计算开销
- 对季节性模式有清晰认知

不适用于：
- 多模态分布（如 GPU 利用率经常在 0% 和 80% 之间跳变）
- 需要检测缓慢漂移
- 指标间的复杂相关性
"""

class StatisticalDetector:
    """统计异常检测器"""

    def z_score_detection(self, series: np.ndarray, window: int = 60,
                          threshold: float = 3.0):
        """
        滑动窗口 Z-score 检测

        优点：简单、快速、可解释
        缺点：假设正态分布，对周期性指标误报多
        """
        rolling_mean = pd.Series(series).rolling(window).mean()
        rolling_std = pd.Series(series).rolling(window).std()

        z_scores = (series - rolling_mean) / rolling_std
        anomalies = np.abs(z_scores) > threshold

        return anomalies, z_scores

    def mad_detection(self, series: np.ndarray, threshold: float = 3.5):
        """
        MAD (Median Absolute Deviation) 检测

        比 Z-score 更鲁棒，不受极端值影响
        适合 GPU 指标这类偶尔有 spike 的数据
        """
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        modified_z = 0.6745 * (series - median) / mad

        return np.abs(modified_z) > threshold

    def seasonal_decomposition(self, series: np.ndarray, period: int = 24):
        """
        季节性分解后的残差检测

        将指标分解为：趋势 + 季节性 + 残差
        只对残差做异常检测，避免将正常的周期波动误报为异常

        适用：GPU 利用率（有明显日周期）、请求量（有工作日/周末周期）
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        result = seasonal_decompose(series, period=period, model='additive')
        residual = result.resid

        # 对残差做 MAD 检测
        anomalies = self.mad_detection(residual[~np.isnan(residual)])

        return anomalies, result

    def ewma_detection(self, series: np.ndarray, span: int = 30,
                       threshold_multiplier: float = 3.0):
        """
        EWMA (Exponentially Weighted Moving Average) 检测

        对最近的数据给更高权重
        适合检测突然的性能退化（如 TTFT 突增）
        """
        ewma = pd.Series(series).ewm(span=span).mean()
        ewm_std = pd.Series(series).ewm(span=span).std()

        upper = ewma + threshold_multiplier * ewm_std
        lower = ewma - threshold_multiplier * ewm_std

        anomalies = (series > upper) | (series < lower)
        return anomalies, {"ewma": ewma, "upper": upper, "lower": lower}
```

### 方法二：机器学习方法

```python
class MLDetector:
    """基于 ML 的异常检测"""

    def isolation_forest(self, metrics_matrix: np.ndarray,
                         contamination: float = 0.01):
        """
        Isolation Forest — 多维异常检测

        核心思想：异常点更容易被"隔离"（需要更少的随机分割）
        
        适用场景：
        - 同时监控多个指标（GPU Util + Memory + Power + Temp）
        - 单个指标正常但组合异常（如 GPU Util 高但 Throughput 低）
        - 不需要标注数据

        输入 metrics_matrix 的列可能是：
        [sm_active, tensor_active, memory_util, power_draw, 
         temperature, pcie_throughput, nvlink_throughput]
        """
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            max_samples='auto',
            random_state=42,
        )

        # 训练（用历史正常数据）
        model.fit(metrics_matrix)

        # 预测（-1 = 异常，1 = 正常）
        predictions = model.predict(metrics_matrix)
        scores = model.decision_function(metrics_matrix)

        return predictions == -1, scores

    def autoencoder_detection(self, metrics_matrix: np.ndarray,
                              threshold_percentile: float = 99):
        """
        Autoencoder 异常检测

        核心思想：训练模型重建正常模式，重建误差大的是异常
        
        优势：
        - 能学习复杂的非线性关系
        - 适合高维数据
        - 重建误差可作为"异常程度"的度量
        """
        import torch
        import torch.nn as nn

        class MetricsAutoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim=8):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, encoding_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, input_dim),
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        # 训练后用重建误差检测异常
        model = MetricsAutoencoder(input_dim=metrics_matrix.shape[1])
        # ... 训练代码省略 ...

        reconstructed = model(torch.FloatTensor(metrics_matrix))
        reconstruction_error = torch.mean(
            (torch.FloatTensor(metrics_matrix) - reconstructed) ** 2, dim=1
        ).detach().numpy()

        threshold = np.percentile(reconstruction_error, threshold_percentile)
        anomalies = reconstruction_error > threshold

        return anomalies, reconstruction_error

    def dbscan_clustering(self, metrics_matrix: np.ndarray, eps: float = 0.5):
        """
        DBSCAN 聚类异常检测

        核心思想：不属于任何簇的点就是异常
        适合发现 GPU 的不同工作模式（idle/inference/training clusters）
        """
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaled = scaler.fit_transform(metrics_matrix)

        clustering = DBSCAN(eps=eps, min_samples=5).fit(scaled)
        # label = -1 表示噪声点（异常）
        anomalies = clustering.labels_ == -1

        return anomalies, clustering.labels_
```

---

## 根因分析（RCA）

### 基于因果图的 RCA

```
GPU 推理服务的因果关系图：

  [NVLink 故障] ──→ [TP 通信延迟↑] ──→ [TTFT↑]
                                      ↗
  [KV Cache 满] ──→ [请求排队] ──→ [响应延迟↑]
         ↑                            ↗
  [流量突增] ──────────────────────────
         │
         └──→ [GPU Util↑] ──→ [温度↑] ──→ [降频] ──→ [吞吐↓]

  [显存泄漏] ──→ [OOM] ──→ [进程重启] ──→ [服务中断]

  [CUDA 错误] ──→ [ECC 报错] ──→ [GPU 掉卡] ──→ [TP 组不可用]
```

### 自动根因分析引擎

```python
class RootCauseAnalyzer:
    """自动根因分析"""

    # 定义因果规则（可配置）
    CAUSAL_RULES = [
        {
            "symptom": "ttft_p99_high",
            "possible_causes": [
                {"cause": "prefill_queue_full", "check": "queue_depth > 50",
                 "confidence": 0.8},
                {"cause": "kv_cache_full", "check": "kv_cache_usage > 95%",
                 "confidence": 0.9},
                {"cause": "nvlink_degraded", "check": "nvlink_bandwidth < 50%_baseline",
                 "confidence": 0.7},
                {"cause": "gpu_throttling", "check": "gpu_clock < 90%_max",
                 "confidence": 0.6},
            ],
        },
        {
            "symptom": "throughput_drop",
            "possible_causes": [
                {"cause": "gpu_failure", "check": "gpu_count < expected",
                 "confidence": 0.95},
                {"cause": "memory_pressure", "check": "memory_util > 95%",
                 "confidence": 0.7},
                {"cause": "batch_size_regression", "check": "avg_batch_size < baseline * 0.5",
                 "confidence": 0.6},
                {"cause": "model_loading", "check": "model_load_in_progress",
                 "confidence": 0.9},
            ],
        },
        {
            "symptom": "error_rate_spike",
            "possible_causes": [
                {"cause": "oom_kills", "check": "oom_events > 0",
                 "confidence": 0.95},
                {"cause": "cuda_error", "check": "xid_errors > 0",
                 "confidence": 0.9},
                {"cause": "timeout_from_overload", "check": "queue_wait > timeout_threshold",
                 "confidence": 0.7},
                {"cause": "bad_input", "check": "input_validation_errors > baseline * 5",
                 "confidence": 0.6},
            ],
        },
    ]

    def analyze(self, alert: dict) -> dict:
        """对一个告警进行根因分析"""

        symptom = alert["metric_name"]
        timestamp = alert["timestamp"]

        # 找到匹配的因果规则
        matching_rules = [r for r in self.CAUSAL_RULES if r["symptom"] == symptom]

        if not matching_rules:
            return {"status": "no_rules_match", "suggestion": "manual_investigation"}

        # 检查每个可能的根因
        findings = []
        for rule in matching_rules:
            for cause in rule["possible_causes"]:
                evidence = self.check_condition(cause["check"], timestamp)
                if evidence["matched"]:
                    findings.append({
                        "cause": cause["cause"],
                        "confidence": cause["confidence"] * evidence["strength"],
                        "evidence": evidence["details"],
                        "remediation": self.get_remediation(cause["cause"]),
                    })

        # 按置信度排序
        findings.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "alert": alert,
            "root_causes": findings[:3],  # 返回 Top 3 可能原因
            "recommended_action": findings[0]["remediation"] if findings else None,
        }

    def correlate_alerts(self, alerts: list, time_window_seconds: int = 300):
        """
        关联在时间窗口内的多个告警，寻找共同根因
        
        例：同时出现 TTFT↑ + GPU Util↑ + Queue Depth↑
        → 很可能是流量突增，而不是三个独立的问题
        """
        # 按时间窗口分组
        alert_groups = self.group_by_time_window(alerts, time_window_seconds)

        for group in alert_groups:
            symptoms = set(a["metric_name"] for a in group)

            # 检查是否匹配已知的复合模式
            for pattern in self.COMPOUND_PATTERNS:
                if symptoms.issuperset(pattern["symptoms"]):
                    return {
                        "pattern": pattern["name"],
                        "root_cause": pattern["root_cause"],
                        "affected_alerts": group,
                        "remediation": pattern["remediation"],
                    }

        return None
```

---

## 自动修复（Auto-Remediation）

### 安全的自动修复框架

```
自动修复的铁律：

1. 只做可逆操作（永远能回滚）
2. 限制爆炸半径（一次只动一个组件）
3. 人在环路中（关键操作需审批）
4. 冷却时间（同一问题短时间内只修复一次）
5. 完整审计（所有操作留记录）
```

```python
class AutoRemediation:
    """自动修复引擎"""

    # 修复动作注册表
    REMEDIATION_ACTIONS = {
        "restart_vllm_instance": {
            "risk": "low",
            "auto_approve": True,
            "cooldown_minutes": 30,
            "max_retries": 2,
            "rollback": "none_needed",  # 重启本身就是恢复
        },
        "scale_up_replicas": {
            "risk": "low",
            "auto_approve": True,
            "cooldown_minutes": 15,
            "max_step": 2,
            "rollback": "scale_down_after_30m",
        },
        "drain_and_replace_gpu": {
            "risk": "medium",
            "auto_approve": False,  # 需要人工确认
            "cooldown_minutes": 60,
            "rollback": "undrain_gpu",
        },
        "failover_to_backup": {
            "risk": "medium",
            "auto_approve": True,  # 故障转移可以自动
            "cooldown_minutes": 5,
            "rollback": "failback_to_primary",
        },
        "clear_kv_cache": {
            "risk": "low",
            "auto_approve": True,
            "cooldown_minutes": 10,
            "rollback": "none_needed",
        },
        "reduce_max_batch_size": {
            "risk": "medium",
            "auto_approve": True,
            "cooldown_minutes": 20,
            "rollback": "restore_batch_size",
        },
    }

    def execute_remediation(self, action_name: str, context: dict):
        """执行修复动作"""

        action = self.REMEDIATION_ACTIONS[action_name]

        # 检查冷却时间
        if self.is_in_cooldown(action_name, context["target"]):
            return {"status": "skipped", "reason": "cooldown_active"}

        # 检查是否需要审批
        if not action["auto_approve"]:
            approval = self.request_approval(action_name, context)
            if not approval["approved"]:
                return {"status": "pending_approval", "ticket": approval["ticket_id"]}

        # 执行前记录状态（用于回滚）
        pre_state = self.capture_state(context["target"])

        # 执行修复
        try:
            result = self.run_action(action_name, context)

            # 验证修复是否生效
            if self.verify_fix(action_name, context):
                self.record_success(action_name, context, pre_state)
                return {"status": "success", "result": result}
            else:
                # 修复未生效，执行回滚
                self.rollback(action["rollback"], pre_state)
                return {"status": "fix_ineffective", "rolled_back": True}

        except Exception as e:
            self.rollback(action["rollback"], pre_state)
            self.alert_oncall(f"Auto-remediation failed: {action_name}", e)
            return {"status": "error", "error": str(e), "rolled_back": True}
```

### 常见自动修复场景

```yaml
# 自动修复场景配置
remediation_playbooks:
  - name: "vLLM OOM Recovery"
    trigger:
      alert: "vllm_oom_error"
      count: ">= 1"
    steps:
      - action: clear_kv_cache
        wait: 10s
      - action: check_health
        expect: healthy
      - action: if_unhealthy_restart_instance
    escalation:
      if_fails: page_oncall

  - name: "High Latency - Queue Buildup"
    trigger:
      alert: "ttft_p99 > slo_target * 1.5"
      duration: "5m"
    steps:
      - action: check_queue_depth
      - action: if_queue_high_scale_up
        max_replicas: current + 2
      - action: wait_for_healthy
        timeout: 300s
    escalation:
      if_fails: reduce_traffic_weight

  - name: "GPU Error Recovery"
    trigger:
      alert: "xid_error_detected"
    steps:
      - action: identify_faulty_gpu
      - action: drain_workload_from_gpu
        grace_period: 60s
      - action: attempt_gpu_reset
      - action: if_reset_fails_mark_unhealthy
    escalation:
      if_fails: create_hardware_ticket
```

---

## ChatOps 集成

### ChatOps 的价值

```
不是所有操作都需要打开终端和 Grafana：

日常运维场景：
  "当前 GPU 利用率怎么样？" → Bot 回复摘要
  "最近有什么异常吗？" → Bot 回复近期 alert 汇总
  "帮我扩容一个推理实例" → Bot 执行并汇报
  "过去一周的成本报告" → Bot 生成并发送

价值：
  1. 降低运维门槛（不用记命令、不用翻 Dashboard）
  2. 操作可审计（所有命令在聊天记录中）
  3. 知识共享（团队都能看到操作过程）
  4. 移动友好（手机上也能运维）
```

### ChatOps Bot 架构

```python
class InfraBot:
    """AI Infra ChatOps Bot"""

    # 命令注册
    COMMANDS = {
        "status": {
            "description": "查看集群状态摘要",
            "permission": "viewer",
            "handler": "handle_status",
        },
        "alerts": {
            "description": "查看当前活跃告警",
            "permission": "viewer",
            "handler": "handle_alerts",
        },
        "scale": {
            "description": "扩缩容操作",
            "permission": "operator",
            "handler": "handle_scale",
            "requires_confirmation": True,
        },
        "cost": {
            "description": "查看成本报告",
            "permission": "viewer",
            "handler": "handle_cost",
        },
        "diagnose": {
            "description": "对指定问题进行诊断",
            "permission": "operator",
            "handler": "handle_diagnose",
        },
        "restart": {
            "description": "重启指定服务",
            "permission": "operator",
            "handler": "handle_restart",
            "requires_confirmation": True,
        },
    }

    async def handle_message(self, message: str, user: str):
        """处理用户消息"""

        # 解析意图（可以用 LLM 理解自然语言）
        intent = await self.parse_intent(message)

        # 权限检查
        if not self.check_permission(user, intent["command"]):
            return f"权限不足：{intent['command']} 需要 {self.COMMANDS[intent['command']]['permission']} 角色"

        # 需要确认的危险操作
        if self.COMMANDS[intent["command"]].get("requires_confirmation"):
            return await self.request_confirmation(intent, user)

        # 执行
        handler = getattr(self, self.COMMANDS[intent["command"]]["handler"])
        result = await handler(intent["params"])

        # 格式化回复
        return self.format_response(result)

    async def handle_status(self, params):
        """集群状态摘要"""
        metrics = await self.collect_cluster_metrics()

        return {
            "gpu_summary": {
                "total": 8,
                "healthy": metrics["healthy_gpus"],
                "utilization_avg": f"{metrics['avg_sm_active']:.1f}%",
                "temperature_max": f"{metrics['max_temp']}°C",
            },
            "inference_summary": {
                "active_models": metrics["model_count"],
                "total_qps": f"{metrics['total_qps']:.1f}",
                "ttft_p99": f"{metrics['ttft_p99_ms']:.0f}ms",
                "error_rate": f"{metrics['error_rate']:.2%}",
            },
            "alerts": {
                "critical": metrics["critical_alerts"],
                "warning": metrics["warning_alerts"],
            },
            "cost_today": f"${metrics['cost_today']:.2f}",
        }

    async def handle_diagnose(self, params):
        """智能诊断"""
        issue = params.get("issue", "")

        # 收集相关指标
        relevant_metrics = await self.gather_diagnostic_data(issue)

        # 用 LLM 分析（Meta: 用 AI 诊断 AI Infra）
        diagnosis = await self.llm_analyze(
            prompt=f"""
            作为 AI Infra 运维专家，分析以下问题：
            问题描述：{issue}
            
            相关指标：
            {json.dumps(relevant_metrics, indent=2)}
            
            请给出：
            1. 最可能的根因（附置信度）
            2. 建议的修复步骤
            3. 需要进一步确认的信息
            """,
            metrics=relevant_metrics,
        )

        return diagnosis
```

### ChatOps 告警整合

```yaml
# 告警 → ChatOps 的智能路由
alert_routing:
  channels:
    - name: "#gpu-alerts-critical"
      filter:
        severity: critical
      format: |
        :rotating_light: *CRITICAL* | {alert_name}
        GPU: {gpu_id} | 持续: {duration}
        根因分析: {auto_rca_result}
        建议操作: {suggested_action}
        [一键修复] [静默 1h] [升级]

    - name: "#gpu-alerts-warning"
      filter:
        severity: warning
      format: |
        :warning: *WARNING* | {alert_name}
        {summary}
        [详情] [静默]

    - name: "#daily-report"
      schedule: "0 9 * * *"  # 每天 9 点
      content: daily_summary
      
  # 智能聚合：5 分钟内的同类告警合并为一条
  aggregation:
    window: 5m
    group_by: [alert_name, gpu_id]
    template: |
      :bell: {alert_name} × {count} 次
      影响: {affected_gpus} 个 GPU
      首次: {first_time} | 最近: {last_time}
```

---

## 预测性维护

### 从被动响应到主动预防

```
运维成熟度模型：

Level 1: 被动响应（Reactive）
  "GPU 挂了 → 修"
  MTTR: 30-60 分钟

Level 2: 主动监控（Proactive）
  "GPU 指标异常 → 告警 → 人工介入"
  MTTR: 10-30 分钟

Level 3: 预测性维护（Predictive）
  "GPU 即将出问题 → 提前迁移 → 无影响"
  MTTR: 接近 0（预防了问题发生）

Level 4: 自愈（Self-healing）
  "GPU 出问题 → 自动检测 → 自动修复 → 自动验证"
  MTTR: 1-5 分钟（全自动）
```

### GPU 故障预测

```python
class GPUFailurePredictor:
    """GPU 故障预测模型"""

    # 故障前兆指标
    PRECURSOR_FEATURES = [
        "ecc_errors_volatile",       # ECC 错误计数
        "retired_pages_count",       # 退役内存页
        "temperature_trend",         # 温度趋势（斜率）
        "power_violation_count",     # 功率违规次数
        "xid_error_frequency",       # XID 错误频率
        "pcie_replay_count",         # PCIe 重传次数
        "memory_clock_throttle",     # 显存时钟降频事件
        "gpu_clock_deviation",       # GPU 时钟偏离标准值
    ]

    def predict_failure_probability(self, gpu_id: int,
                                     horizon_hours: int = 72):
        """预测 GPU 在未来 N 小时内故障的概率"""

        features = self.extract_features(gpu_id)

        # 使用预训练的 Gradient Boosting 模型
        probability = self.model.predict_proba(features.reshape(1, -1))[0][1]

        # 基于概率的行动建议
        if probability > 0.8:
            action = "immediate_drain_and_replace"
        elif probability > 0.5:
            action = "schedule_maintenance_window"
        elif probability > 0.3:
            action = "increase_monitoring_frequency"
        else:
            action = "no_action"

        return {
            "gpu_id": gpu_id,
            "failure_probability": probability,
            "horizon_hours": horizon_hours,
            "top_risk_factors": self.explain_prediction(features),
            "recommended_action": action,
        }
```

---

## AIOps 成熟度路线图

```
你的 8 张 H20 集群 AIOps 建设路线：

Phase 1（立即可做，1-2 周）：
  □ 统计异常检测（EWMA + MAD）
  □ 基于规则的自动修复（重启、扩容）
  □ ChatOps 基础命令（status、alerts）
  □ 告警聚合减噪

Phase 2（短期，1-2 个月）：
  □ 多维异常检测（Isolation Forest）
  □ 基于因果图的根因分析
  □ 预测性维护（ECC 错误趋势）
  □ 成本异常检测

Phase 3（中期，3-6 个月）：
  □ Autoencoder 深度异常检测
  □ LLM 驱动的智能诊断
  □ 全自动修复闭环
  □ 容量预测 + 自动采购建议

目标：达到 Level 3-4 运维成熟度
  - 90% 的常见问题自动修复
  - MTTR < 5 分钟
  - 误报率 < 5%
  - 每周运维时间从 10h 降到 2h
```

---

## 下一步

→ 进入 [labs/](../labs/) 开始动手实践
