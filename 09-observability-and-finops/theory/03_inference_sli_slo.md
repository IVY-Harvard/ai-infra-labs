# 03 - 推理服务 SLI/SLO 设计

## 为什么推理服务需要专门的 SLI/SLO

### 传统 Web 服务 vs 推理服务

```
传统 Web 服务：
  - 延迟基本可预测（固定逻辑路径）
  - P99/P50 比值通常 2-5x
  - 资源消耗相对均匀
  - 容量弹性扩缩容简单

推理服务：
  - 延迟取决于输入/输出长度（不可预测）
  - P99/P50 比值可达 10-50x
  - 资源消耗随 batch 状态剧烈波动
  - 容量受限于 GPU（扩容周期长、成本高）
```

### SLI/SLO/SLA 三层体系

```
SLA (Service Level Agreement)
  └── 合同承诺，有赔偿条款
  └── 例：月度可用性 >= 99.9%，否则赔偿 10% 月费

SLO (Service Level Objective)
  └── 内部目标，比 SLA 更严格
  └── 例：TTFT P99 < 500ms (内部目标 < 300ms)

SLI (Service Level Indicator)
  └── 实际测量值
  └── 例：当前 TTFT P99 = 245ms
```

---

## 推理服务核心 SLI 定义

### 1. TTFT — Time To First Token

```
定义：从客户端发送请求到收到第一个生成 token 的时间
测量点：客户端视角（包含网络延迟）或服务端视角（不含）

TTFT = 网络延迟 + 排队时间 + Prefill 时间

分解：
  网络延迟：通常 1-50ms（取决于部署位置）
  排队时间：0ms - 数秒（取决于并发量和 GPU 资源）
  Prefill 时间：取决于 prompt 长度和模型大小

影响因素：
  - Prompt 长度（token 数量）
  - 模型大小（参数量）
  - 当前 batch size
  - KV Cache Prefix 命中率
  - GPU 型号（计算能力）
```

**TTFT 的 SLO 参考值（H20 + 70B 模型）**：

```
场景: 在线对话（用户在等待）
  P50 < 200ms
  P95 < 500ms
  P99 < 1000ms

场景: API 调用（异步处理）
  P50 < 500ms
  P95 < 2000ms
  P99 < 5000ms

场景: 批量处理（离线任务）
  P50 < 2000ms
  P95 < 10000ms
  P99 < 30000ms
```

### 2. TPOT — Time Per Output Token

```
定义：Decode 阶段每生成一个 token 的平均时间
也称：Inter-Token Latency (ITL)

TPOT = Decode 总时间 / 生成 token 数量

影响因素：
  - 当前 batch size（continuous batching 下动态变化）
  - KV Cache 大小（所有 batch 中请求的 KV Cache 总和）
  - 模型大小和量化方式
  - HBM 带宽（H20 = 4TB/s，Decode 是 memory-bound）
```

**TPOT 的 SLO 参考值（H20 + 70B 模型）**：

```
场景: 在线对话（用户在阅读流式输出）
  P50 < 30ms/token (~33 tokens/s，人类阅读速度足够)
  P95 < 60ms/token
  P99 < 100ms/token

场景: API 调用
  P50 < 50ms/token
  P95 < 100ms/token
  P99 < 200ms/token
```

### 3. End-to-End Latency

```
定义：完整请求从发出到最后一个 token 收到的总时间

E2E Latency = TTFT + TPOT × output_token_count

注意：这个指标受输出长度影响巨大，通常按输出长度分桶统计
```

### 4. Throughput

```
定义：系统级吞吐量，多种计量方式

指标：
  - tokens_per_second（生成 token 速率，最常用）
  - requests_per_second（请求完成速率）
  - prefill_tokens_per_second（Prefill 处理速率）

分维度：
  - 单 GPU 吞吐
  - 单实例（可能多 GPU TP/PP）吞吐
  - 集群总吞吐
```

**吞吐量 SLO 参考值（8×H20, TP=8, 70B 模型）**：

```
tokens_per_second(generation) >= 2000 tokens/s  (整体)
tokens_per_second(generation) >= 250 tokens/s   (单 GPU)
requests_per_second >= 20 req/s                  (batch_size=1 等效)
```

### 5. KV Cache Hit Rate（Prefix Caching）

```
定义：请求命中前缀缓存的比例

kv_cache_hit_rate = prefix_matched_tokens / total_prompt_tokens

意义：
  - 高命中率 → TTFT 大幅降低
  - 常见于系统 prompt 相同的场景
  - 直接影响成本（减少 Prefill 计算量）
```

### 6. Queue Wait Time

```
定义：请求在调度队列中等待的时间

queue_wait_time = 开始 Prefill 时间 - 请求到达时间

SLO 参考：
  在线场景：P99 < 100ms
  API 场景：P99 < 1000ms
  批处理：P99 < 60000ms

告警阈值：
  Warning: P99 > SLO × 0.8
  Critical: P99 > SLO
```

### 7. Error Rate

```
定义：请求失败率

error_rate = failed_requests / total_requests

错误类型：
  - 429 Too Many Requests（限流）
  - 503 Service Unavailable（GPU 资源不足）
  - 500 Internal Error（CUDA OOM, 模型错误等）
  - Timeout（请求超时）

SLO 参考：
  error_rate < 0.1% (P99 时段)
  error_rate < 0.01% (正常时段)
```

---

## SLO 设计方法论

### Step 1: 定义用户体验目标

```
用户类型          期望体验              对应 SLI
─────────────────────────────────────────────────────
对话用户          "感觉很快"           TTFT P95 < 500ms
对话用户          "生成很流畅"         TPOT P95 < 50ms
API 开发者        "调用可靠"           Error Rate < 0.1%
API 开发者        "延迟可预测"         E2E P99/P50 < 5x
批量处理          "吞吐足够"           tokens/s > 阈值
内部平台团队      "成本可控"           cost/token < 预算
```

### Step 2: 基于 SLI 确定 SLO

**关键原则：SLO 是 SLA 的安全边际**

```python
# SLO 设计公式
SLO = SLA_target × safety_margin

# 例：SLA 承诺 TTFT P99 < 1000ms
# safety_margin = 0.6（60% 裕度）
SLO_TTFT_P99 = 1000 * 0.6 = 600  # ms

# Error Budget = 1 - SLO
# 如果 SLO = 99.9%，Error Budget = 0.1%
# 一个月 = 43200 分钟，可以有 43.2 分钟的故障时间
```

### Step 3: 多维度 SLO 矩阵

```yaml
slo_matrix:
  tier_1_interactive:  # 面向终端用户的实时对话
    ttft_p50_ms: 200
    ttft_p99_ms: 800
    tpot_p50_ms: 25
    tpot_p99_ms: 80
    error_rate: 0.001
    availability: 0.999
    queue_wait_p99_ms: 100

  tier_2_api:  # 面向开发者的 API
    ttft_p50_ms: 500
    ttft_p99_ms: 2000
    tpot_p50_ms: 40
    tpot_p99_ms: 150
    error_rate: 0.005
    availability: 0.995
    queue_wait_p99_ms: 1000

  tier_3_batch:  # 批量处理任务
    throughput_tokens_per_second: 5000
    error_rate: 0.01
    completion_rate_24h: 0.99
    availability: 0.99
```

### Step 4: SLO 监控和 Error Budget

```python
# Error Budget Policy
class ErrorBudgetPolicy:
    """
    Error Budget 策略：
    - Budget > 50%: 正常迭代，可以做变更
    - Budget 25-50%: 谨慎变更，需要 review
    - Budget < 25%: 冻结变更，只修 bug
    - Budget < 0%: 紧急响应，所有人修 reliability
    """

    def evaluate(self, slo_target: float, current_good_ratio: float,
                 window_days: int = 30) -> str:
        budget_total = 1 - slo_target  # 例：0.001 for 99.9%
        budget_used = 1 - current_good_ratio
        budget_remaining = (budget_total - budget_used) / budget_total

        if budget_remaining > 0.5:
            return "GREEN - 正常迭代"
        elif budget_remaining > 0.25:
            return "YELLOW - 谨慎变更"
        elif budget_remaining > 0:
            return "ORANGE - 冻结变更"
        else:
            return "RED - 紧急响应"
```

---

## SLA 合同怎么写

### SLA 合同核心条款模板

```markdown
## 服务等级协议 (SLA)

### 1. 服务可用性
- 月度可用性 >= 99.9%（约 43 分钟/月停机时间）
- 可用性 = (总分钟数 - 不可用分钟数) / 总分钟数
- 计划维护窗口不计入不可用时间（需提前 72 小时通知）

### 2. 性能指标
| 指标 | 标准 | 测量方式 |
|------|------|---------|
| TTFT (P99) | < 1000ms | 服务端测量 |
| TPOT (P99) | < 100ms | 服务端测量 |
| 错误率 | < 0.5% | 5xx / total |
| API 响应时间 (P99) | < 30s | 端到端 |

### 3. 赔偿条款
| 月度可用性 | 服务信用额度 |
|-----------|-------------|
| 99.0% - 99.9% | 月费 10% |
| 95.0% - 99.0% | 月费 25% |
| < 95.0% | 月费 50% |

### 4. 排除条款
以下情况不计入 SLA 违规：
- 客户端原因导致的请求失败
- 超过合同约定并发数的请求被限流
- 不可抗力（自然灾害、政策变更等）
- 计划维护期间的服务中断
- 因客户配置错误导致的问题

### 5. SLA 报告
- 提供月度 SLA 合规报告
- 包含：可用性数据、延迟分位数、错误率统计
- 报告在次月第 5 个工作日前提供
```

### 关键陷阱

```
陷阱 1: 测量点不一致
  ✗ 客户端测量的 TTFT（包含网络延迟）
  ✓ 服务端测量的 TTFT（可控因素）
  → SLA 中必须明确测量点

陷阱 2: 平均值 vs 分位数
  ✗ "平均 TTFT < 200ms"（平均值隐藏了尾部延迟）
  ✓ "TTFT P99 < 1000ms"（保护绝大多数请求）
  → SLA 中应该用分位数

陷阱 3: 粒度太粗
  ✗ "月度请求成功率 > 99.9%"
  → 月初出了 1 小时全量故障，剩下的时间 100%，平均值达标
  ✓ "任意滚动 5 分钟窗口的成功率 > 99%"
  → 能捕捉短时故障

陷阱 4: 没有区分请求类型
  ✗ "所有请求 TTFT P99 < 500ms"
  → 一个 100K token 的 prompt 也要 500ms？
  ✓ 按 prompt 长度分桶定义 SLO
  → "prompt < 1K tokens: TTFT P99 < 500ms"
  → "prompt 1K-10K tokens: TTFT P99 < 2000ms"
```

---

## Prometheus 中实现 SLI/SLO 监控

### Recording Rules

```yaml
groups:
  - name: inference_sli
    interval: 15s
    rules:
      # TTFT 分位数
      - record: inference:ttft_seconds:p50
        expr: histogram_quantile(0.50, sum(rate(vllm_e2e_request_latency_seconds_bucket[5m])) by (le))

      - record: inference:ttft_seconds:p99
        expr: histogram_quantile(0.99, sum(rate(vllm_e2e_request_latency_seconds_bucket[5m])) by (le))

      # SLO 合规率
      - record: inference:ttft_slo_compliance:ratio
        expr: |
          sum(rate(vllm_e2e_request_latency_seconds_bucket{le="1.0"}[30m]))
          /
          sum(rate(vllm_e2e_request_latency_seconds_count[30m]))

      # Error Budget 剩余
      - record: inference:error_budget_remaining:ratio
        expr: |
          1 - (
            (1 - inference:ttft_slo_compliance:ratio)
            /
            (1 - 0.999)  # SLO target = 99.9%
          )
```

### Alerting Rules

```yaml
groups:
  - name: slo_alerts
    rules:
      # Burn rate alert（SLO 消耗速度告警）
      - alert: HighErrorBudgetBurnRate
        expr: |
          (
            inference:error_budget_remaining:ratio < 0.5
            and
            # 最近 1 小时消耗速度 > 14.4x（1 天耗完 budget 的速度）
            (1 - inference:ttft_slo_compliance:ratio) > 14.4 * (1 - 0.999)
          )
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "SLO Error Budget 正在快速消耗"
          description: "当前 Error Budget 剩余 {{ $value | humanizePercentage }}，按当前速度将在 24h 内耗尽"

      # TTFT P99 超限
      - alert: TTFTHighLatency
        expr: inference:ttft_seconds:p99 > 0.8  # SLO 的 80% 作为预警
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "TTFT P99 接近 SLO 阈值"

      # 吞吐量下降
      - alert: ThroughputDrop
        expr: |
          rate(vllm_generation_tokens_total[5m]) < 0.7 *
          avg_over_time(rate(vllm_generation_tokens_total[5m])[1h:5m])
        for: 10m
        labels:
          severity: warning
```

---

## 按 Prompt 长度分桶的高级 SLO

```python
# 推理服务侧打标签
@app.middleware
async def label_request_by_prompt_length(request, call_next):
    prompt_tokens = count_tokens(request.prompt)

    # 分桶
    if prompt_tokens < 1000:
        bucket = "short"
    elif prompt_tokens < 10000:
        bucket = "medium"
    elif prompt_tokens < 50000:
        bucket = "long"
    else:
        bucket = "very_long"

    # 将 bucket 作为标签传给 metrics
    with metrics_context(prompt_length_bucket=bucket):
        response = await call_next(request)

    return response
```

```yaml
# 按桶定义 SLO
slo_by_prompt_length:
  short:      # < 1K tokens
    ttft_p99_ms: 500
    tpot_p99_ms: 60
  medium:     # 1K - 10K tokens
    ttft_p99_ms: 2000
    tpot_p99_ms: 80
  long:       # 10K - 50K tokens
    ttft_p99_ms: 5000
    tpot_p99_ms: 100
  very_long:  # > 50K tokens
    ttft_p99_ms: 15000
    tpot_p99_ms: 120
```

---

## 下一步

→ 进入 [04_distributed_tracing_ai.md](04_distributed_tracing_ai.md) 了解如何追踪一个推理请求的全链路
