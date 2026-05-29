# AI 合规治理框架

## 1. 中国《生成式人工智能服务管理暂行办法》

### 1.1 核心要求

```
发布时间：2023 年 7 月 13 日，8 月 15 日起施行
适用范围：面向中国境内公众提供生成式 AI 服务

关键条款：
┌──────────────────────────────────────────────────┐
│  第四条  提供和使用生成式 AI 服务，应当：          │
│  (1) 坚持社会主义核心价值观                       │
│  (2) 不得生成违法内容                             │
│  (3) 防止歧视                                    │
│  (4) 尊重知识产权                                │
│  (5) 防止虚假信息                                │
├──────────────────────────────────────────────────┤
│  第七条  训练数据要求：                           │
│  (1) 使用合法来源的数据                           │
│  (2) 不侵犯知识产权                              │
│  (3) 涉及个人信息需取得同意或符合法规             │
│  (4) 采取措施提高训练数据质量                     │
├──────────────────────────────────────────────────┤
│  第九条  内容标识：                               │
│  AI 生成的内容应进行标识（水印）                   │
├──────────────────────────────────────────────────┤
│  第十四条 安全评估：                              │
│  具有舆论属性或社会动员能力的服务                  │
│  需按规定开展安全评估并备案                        │
└──────────────────────────────────────────────────┘
```

### 1.2 对企业 AI 系统的影响

```
企业内部使用 vs 对外服务：
┌──────────────────┬──────────────────────────────┐
│  企业内部使用    │  合规要求相对宽松              │
│  (不对公众提供)  │  仍需遵守数据安全/个人信息保护  │
├──────────────────┼──────────────────────────────┤
│  对外提供服务    │  完整合规要求                  │
│  (面向公众)      │  安全评估 + 算法备案           │
│                  │  内容过滤 + 标识 + 投诉机制    │
└──────────────────┴──────────────────────────────┘

技术实施要点：
1. 内容安全过滤（护栏系统）
2. AI 生成内容标识（水印）
3. 用户投诉处理机制
4. 训练数据审计
5. 安全评估报告
```

## 2. 欧盟 AI 法案（EU AI Act）

### 2.1 风险分级体系

```
EU AI Act 四级风险分类：

┌─────────────────────────────────────────────────┐
│  不可接受风险（禁止）                             │
│  - 社会评分系统                                  │
│  - 利用人类弱点的 AI                             │
│  - 实时远程生物识别（执法除外）                   │
├─────────────────────────────────────────────────┤
│  高风险                                          │
│  - 关键基础设施                                  │
│  - 教育和职业培训                                │
│  - 就业和人力资源管理                            │
│  - 信用评估                                     │
│  - 司法和民主程序                                │
│  要求：合规评估、风险管理、透明度、人工监督       │
├─────────────────────────────────────────────────┤
│  有限风险（透明度义务）                           │
│  - 聊天机器人：告知用户在与 AI 交互              │
│  - 深度伪造：标识为 AI 生成                      │
│  - 通用 AI 模型：技术文档 + 版权合规             │
├─────────────────────────────────────────────────┤
│  最小风险（无特殊要求）                           │
│  - AI 游戏                                      │
│  - 垃圾邮件过滤                                  │
└─────────────────────────────────────────────────┘
```

### 2.2 通用 AI 模型（GPAI）要求

```
针对基础模型/通用大模型的特殊要求：

所有 GPAI 模型：
- 维护技术文档
- 遵守版权法（训练数据透明度）
- 提供模型能力和限制的摘要

具有系统性风险的 GPAI（如 GPT-4 级别）：
- 模型评估和对抗测试
- 追踪和报告严重事件
- 确保充分的网络安全保护
- 报告能源消耗

罚款：
- 违反禁止条款：最高 3500 万欧元或全球营收 7%
- 违反其他条款：最高 1500 万欧元或全球营收 3%
```

## 3. 美国 AI 行政令及监管

### 3.1 行政令（EO 14110, 2023）

```
美国 AI 安全行政令核心要点：

1. 安全标准
   - 强大 AI 系统需报告安全测试结果
   - 开发双重用途基础模型需通知政府
   - NIST 制定 AI 安全标准

2. 阈值定义
   - 训练计算量 > 10^26 FLOP（通用模型）
   - 训练计算量 > 10^23 FLOP（生物序列模型）
   需要向商务部报告

3. 红队测试
   - 大型 AI 系统部署前须进行红队测试
   - 结果需与政府共享

4. 水印要求
   - 推动 AI 生成内容的认证和水印标准
   - 商务部开发内容来源认证工具

注意：2025 年政策可能有变化，需跟进最新动态
```

### 3.2 行业自律框架

```
美国 AI 安全的多层治理：
┌────────────────────────────────────────────────┐
│  联邦层面：行政令 + NIST 框架                    │
│  州层面：加州 SB-1047（AI 安全法案）等           │
│  行业层面：前沿 AI 安全承诺                      │
│  企业层面：负责任 AI 政策                        │
└────────────────────────────────────────────────┘
```

## 4. 企业 AI 治理框架设计

### 4.1 四层治理模型

```
企业 AI 治理架构：

┌──────────────────────────────────────────────────┐
│  Layer 1: Policy（策略层）                        │
│  ┌──────────────────────────────────────────┐   │
│  │ - AI 使用政策                            │   │
│  │ - 可接受使用范围                          │   │
│  │ - 风险容忍度定义                          │   │
│  │ - 合规基线要求                            │   │
│  └──────────────────────────────────────────┘   │
├──────────────────────────────────────────────────┤
│  Layer 2: Process（流程层）                       │
│  ┌──────────────────────────────────────────┐   │
│  │ - AI 系统上线审批流程                     │   │
│  │ - 风险评估流程                            │   │
│  │ - 事件响应流程                            │   │
│  │ - 模型变更管理流程                        │   │
│  └──────────────────────────────────────────┘   │
├──────────────────────────────────────────────────┤
│  Layer 3: Tools（工具层）                         │
│  ┌──────────────────────────────────────────┐   │
│  │ - 护栏系统（NeMo/Llama Guard）           │   │
│  │ - 审计日志系统                            │   │
│  │ - 访问控制系统（RBAC）                    │   │
│  │ - 监控告警系统                            │   │
│  └──────────────────────────────────────────┘   │
├──────────────────────────────────────────────────┤
│  Layer 4: Audit（审计层）                         │
│  ┌──────────────────────────────────────────┐   │
│  │ - 合规检查报告                            │   │
│  │ - 模型行为审计                            │   │
│  │ - 数据使用审计                            │   │
│  │ - 定期安全评估                            │   │
│  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

### 4.2 风险评估框架

```python
from dataclasses import dataclass
from enum import Enum
from typing import List

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskCategory(Enum):
    SAFETY = "safety"           # 安全风险
    PRIVACY = "privacy"         # 隐私风险
    FAIRNESS = "fairness"       # 公平性风险
    RELIABILITY = "reliability" # 可靠性风险
    COMPLIANCE = "compliance"   # 合规风险

@dataclass
class RiskAssessment:
    """AI 系统风险评估"""
    system_name: str
    description: str
    risk_items: List[dict]
    overall_risk: RiskLevel
    mitigation_plan: str
    reviewer: str
    review_date: str

class AIRiskAssessor:
    """AI 风险评估工具"""
    
    RISK_MATRIX = {
        # (影响程度, 发生概率) → 风险等级
        ("high", "high"): RiskLevel.CRITICAL,
        ("high", "medium"): RiskLevel.HIGH,
        ("high", "low"): RiskLevel.MEDIUM,
        ("medium", "high"): RiskLevel.HIGH,
        ("medium", "medium"): RiskLevel.MEDIUM,
        ("medium", "low"): RiskLevel.LOW,
        ("low", "high"): RiskLevel.MEDIUM,
        ("low", "medium"): RiskLevel.LOW,
        ("low", "low"): RiskLevel.LOW,
    }
    
    def assess(self, system_info: dict) -> RiskAssessment:
        """执行风险评估"""
        risk_items = []
        
        # 评估各类风险
        risk_items.append(self._assess_safety(system_info))
        risk_items.append(self._assess_privacy(system_info))
        risk_items.append(self._assess_fairness(system_info))
        risk_items.append(self._assess_compliance(system_info))
        
        # 计算整体风险
        overall = self._calculate_overall_risk(risk_items)
        
        return RiskAssessment(
            system_name=system_info["name"],
            description=system_info["description"],
            risk_items=risk_items,
            overall_risk=overall,
            mitigation_plan=self._generate_mitigation(risk_items),
            reviewer="",
            review_date=""
        )
    
    def _assess_safety(self, info: dict) -> dict:
        """安全风险评估"""
        risk = {"category": "safety", "items": []}
        
        # 检查是否有护栏
        if not info.get("has_guardrails"):
            risk["items"].append({
                "issue": "未部署护栏系统",
                "impact": "high",
                "likelihood": "high",
                "level": RiskLevel.CRITICAL
            })
        
        # 检查是否有输入验证
        if not info.get("input_validation"):
            risk["items"].append({
                "issue": "缺少输入验证",
                "impact": "high",
                "likelihood": "medium",
                "level": RiskLevel.HIGH
            })
        
        return risk
    
    def _assess_privacy(self, info: dict) -> dict:
        """隐私风险评估"""
        risk = {"category": "privacy", "items": []}
        
        if info.get("processes_pii"):
            if not info.get("pii_protection"):
                risk["items"].append({
                    "issue": "处理 PII 但未部署保护措施",
                    "impact": "high",
                    "likelihood": "high",
                    "level": RiskLevel.CRITICAL
                })
        
        return risk
    
    def _assess_fairness(self, info: dict) -> dict:
        """公平性风险评估"""
        return {"category": "fairness", "items": []}
    
    def _assess_compliance(self, info: dict) -> dict:
        """合规风险评估"""
        risk = {"category": "compliance", "items": []}
        
        if info.get("serves_public") and not info.get("has_filing"):
            risk["items"].append({
                "issue": "面向公众服务但未完成备案",
                "impact": "high",
                "likelihood": "high",
                "level": RiskLevel.CRITICAL
            })
        
        return risk
    
    def _calculate_overall_risk(self, risk_items: list) -> RiskLevel:
        """取所有风险项中的最高等级"""
        max_level = RiskLevel.LOW
        for category in risk_items:
            for item in category.get("items", []):
                if item["level"].value > max_level.value:
                    max_level = item["level"]
        return max_level
    
    def _generate_mitigation(self, risk_items: list) -> str:
        """生成缓解建议"""
        suggestions = []
        for category in risk_items:
            for item in category.get("items", []):
                if item["level"] in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                    suggestions.append(f"- 解决: {item['issue']}")
        return "\n".join(suggestions) if suggestions else "无高风险项"
```

### 4.3 治理组织架构

```
AI 治理委员会组织结构：

┌─────────────────────────────────────────┐
│           AI 治理委员会                   │
│  (CTO/CISO/法务/业务负责人)              │
└──────────────────┬──────────────────────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
┌──────▼──┐  ┌────▼────┐  ┌──▼──────┐
│ AI 安全  │  │ AI 伦理  │  │ AI 合规  │
│ 工作组   │  │ 工作组   │  │ 工作组   │
│          │  │          │  │          │
│- 红队测试│  │- 公平性  │  │- 法规跟踪│
│- 漏洞管理│  │- 透明度  │  │- 备案管理│
│- 应急响应│  │- 可解释性│  │- 审计报告│
└──────────┘  └──────────┘  └──────────┘
```

## 5. AI 系统上线审批流程

### 5.1 流程定义

```
AI 系统上线标准流程：

  ┌──────────┐
  │ 1. 申请   │ → 填写系统信息表
  └────┬─────┘
       │
  ┌────▼─────┐
  │ 2. 风险   │ → 自动化风险评估
  │    评估   │    + 人工审核
  └────┬─────┘
       │
  ┌────▼─────┐
  │ 3. 安全   │ → 红队测试
  │    测试   │    + 护栏验证
  └────┬─────┘
       │
  ┌────▼─────┐
  │ 4. 合规   │ → 法律审查
  │    审查   │    + 隐私影响评估
  └────┬─────┘
       │
  ┌────▼─────┐
  │ 5. 审批   │ → 根据风险等级
  │    决定   │    确定审批层级
  └────┬─────┘
       │
  ┌────▼─────┐
  │ 6. 部署   │ → 灰度发布
  │    上线   │    + 监控就绪
  └────┬─────┘
       │
  ┌────▼─────┐
  │ 7. 持续   │ → 定期复审
  │    监控   │    + 事件响应
  └──────────┘
```

### 5.2 审批矩阵

```
根据风险等级确定审批权限：

┌──────────┬──────────────┬──────────────────────┐
│ 风险等级  │  审批权限     │  额外要求             │
├──────────┼──────────────┼──────────────────────┤
│ Low      │ 团队负责人    │ 标准安全检查          │
├──────────┼──────────────┼──────────────────────┤
│ Medium   │ 部门总监      │ 安全测试 + 隐私评估   │
├──────────┼──────────────┼──────────────────────┤
│ High     │ VP/CISO      │ 红队测试 + 合规审查    │
├──────────┼──────────────┼──────────────────────┤
│ Critical │ AI 治理委员会  │ 完整评估 + 持续监控   │
│          │              │ + 应急预案             │
└──────────┴──────────────┴──────────────────────┘
```

## 6. 合规检查清单

### 6.1 中国市场合规

```
面向中国市场的 AI 合规检查清单：

□ 算法备案
  □ 完成算法备案（如面向公众）
  □ 安全评估报告
  □ 定期更新备案信息

□ 内容安全
  □ 部署内容过滤/护栏系统
  □ 建立人工审核机制
  □ 实现用户投诉处理
  □ AI 生成内容标识（水印）

□ 数据合规
  □ 训练数据来源合法性审查
  □ 个人信息保护影响评估
  □ 数据跨境传输评估（如适用）
  □ 用户同意机制

□ 安全保障
  □ 网络安全等级保护（等保）
  □ 安全事件应急预案
  □ 定期安全审计

□ 用户权益
  □ 服务协议和隐私政策
  □ 用户知情权保障
  □ 自动化决策的解释机制
```

### 6.2 多法域合规映射

```python
class ComplianceMapper:
    """多法域合规要求映射"""
    
    REQUIREMENTS = {
        "china": {
            "content_filtering": {"required": True, "priority": "critical"},
            "watermark": {"required": True, "priority": "high"},
            "algorithm_filing": {"required": True, "priority": "critical",
                                "condition": "public_facing"},
            "security_assessment": {"required": True, "priority": "high"},
            "data_localization": {"required": True, "priority": "high"},
            "pii_consent": {"required": True, "priority": "critical"},
        },
        "eu": {
            "risk_assessment": {"required": True, "priority": "critical"},
            "transparency": {"required": True, "priority": "high"},
            "human_oversight": {"required": True, "priority": "high",
                               "condition": "high_risk"},
            "data_governance": {"required": True, "priority": "critical"},
            "technical_documentation": {"required": True, "priority": "high"},
            "conformity_assessment": {"required": True, "priority": "critical",
                                     "condition": "high_risk"},
        },
        "us": {
            "red_teaming": {"required": True, "priority": "high",
                           "condition": "frontier_model"},
            "watermark": {"required": False, "priority": "medium"},
            "safety_testing": {"required": True, "priority": "high"},
            "incident_reporting": {"required": True, "priority": "medium"},
        },
    }
    
    def get_requirements(self, jurisdictions: list, system_info: dict) -> dict:
        """获取适用的合规要求"""
        applicable = {}
        for jurisdiction in jurisdictions:
            reqs = self.REQUIREMENTS.get(jurisdiction, {})
            for req_name, req_info in reqs.items():
                condition = req_info.get("condition")
                if condition and not system_info.get(condition):
                    continue
                key = f"{jurisdiction}:{req_name}"
                applicable[key] = req_info
        return applicable
    
    def generate_gap_analysis(self, requirements: dict, 
                              current_state: dict) -> list:
        """生成合规差距分析"""
        gaps = []
        for req_key, req_info in requirements.items():
            if not current_state.get(req_key.split(":")[-1]):
                gaps.append({
                    "requirement": req_key,
                    "priority": req_info["priority"],
                    "status": "not_implemented"
                })
        return sorted(gaps, key=lambda g: 
                     {"critical": 0, "high": 1, "medium": 2, "low": 3}
                     [g["priority"]])
```

## 7. 小结

AI 合规治理关键要点：

1. **了解适用法规**：根据服务范围确定需要遵守的法规
2. **风险分级管理**：不同风险等级采用不同管控强度
3. **建立治理组织**：跨部门协作的治理委员会
4. **流程制度化**：标准化的上线审批和持续监控流程
5. **工具自动化**：合规检查尽可能自动化，降低人工负担
6. **持续跟踪**：法规快速迭代，需要持续关注更新

实践建议：
- 先做好中国市场合规（如在中国运营）
- 采用"合规即代码"理念，将合规要求编码为可执行检查
- 建立合规知识库，跟踪法规变化
- 定期内部审计，不等外部检查
