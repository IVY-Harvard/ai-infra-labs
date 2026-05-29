# 模块 10：AI 安全与治理

## 模块概述

本模块面向有后端开发经验、但对 AI 安全领域不熟悉的工程师，系统讲解 AI 系统（特别是大语言模型）面临的安全威胁、防御手段、隐私保护和合规治理。

通过本模块的学习，读者将掌握：
- AI 安全威胁全景（OWASP LLM Top 10）
- 提示词注入的攻击与防御
- AI 护栏系统的设计与实现
- 数据隐私保护（PII 检测、差分隐私、联邦学习）
- AI 系统的访问控制与审计
- 模型安全（水印、对抗攻击、机密计算）
- 国内外 AI 合规框架

## 前置要求

| 要求 | 说明 |
|------|------|
| 硬件 | 8 张 H20 GPU（部分 Lab 需要 GPU 进行模型推理） |
| 语言 | Python 3.10+，熟悉后端开发 |
| 框架 | 了解 FastAPI / Flask，熟悉 Docker |
| AI 基础 | 了解 LLM 基本原理（Transformer、推理、微调） |

## 目录结构

```
10-ai-security-and-governance/
├── README.md                          # 本文件
├── theory/                            # 理论知识
│   ├── 01_ai_security_landscape.md    # AI 安全全景
│   ├── 02_prompt_injection.md         # 提示词注入深度解析
│   ├── 03_guardrails_systems.md       # AI 护栏系统
│   ├── 04_data_privacy.md             # 数据隐私
│   ├── 05_access_control.md           # 访问控制
│   ├── 06_model_security.md           # 模型安全
│   └── 07_compliance_framework.md     # 合规框架
├── labs/                              # 动手实验
│   ├── 01_prompt_injection/           # 提示词注入攻防
│   ├── 02_guardrails_nemo/            # NeMo Guardrails
│   ├── 03_llama_guard/                # Llama Guard
│   ├── 04_rbac_design/                # RBAC 设计
│   ├── 05_data_privacy/               # 数据隐私
│   ├── 06_model_watermark/            # 模型水印
│   ├── 07_adversarial_attack/         # 对抗攻击
│   ├── 08_confidential_computing/     # 机密计算
│   ├── 09_audit_logging/              # 审计日志
│   └── 10_compliance_framework/       # 合规框架
└── project/
    └── ai-gateway-with-guardrails/    # 企业级项目：AI API 网关
```

## 学习路线

### 第一阶段：威胁认知（2 天）
1. 阅读 `theory/01_ai_security_landscape.md`，建立全局视野
2. 深入 `theory/02_prompt_injection.md`，理解最常见的 LLM 攻击
3. 完成 `labs/01_prompt_injection/`，亲手体验攻击与防御

### 第二阶段：防御体系（3 天）
4. 学习 `theory/03_guardrails_systems.md`，了解护栏方案
5. 完成 `labs/02_guardrails_nemo/` 和 `labs/03_llama_guard/`
6. 学习 `theory/04_data_privacy.md`，完成 `labs/05_data_privacy/`
7. 学习 `theory/05_access_control.md`，完成 `labs/04_rbac_design/`

### 第三阶段：高级安全（2 天）
8. 学习 `theory/06_model_security.md`
9. 完成 `labs/06_model_watermark/` 和 `labs/07_adversarial_attack/`
10. 了解 `labs/08_confidential_computing/`

### 第四阶段：治理与合规（2 天）
11. 学习 `theory/07_compliance_framework.md`
12. 完成 `labs/09_audit_logging/` 和 `labs/10_compliance_framework/`

### 第五阶段：综合项目（3 天）
13. 完成 `project/ai-gateway-with-guardrails/`，构建企业级 AI 网关

## 核心概念速查

| 概念 | 说明 | 对应章节 |
|------|------|----------|
| Prompt Injection | 通过精心构造的输入操纵 LLM 行为 | Theory 02, Lab 01 |
| Guardrails | 在 LLM 输入/输出端设置的安全屏障 | Theory 03, Lab 02-03 |
| PII Detection | 识别并保护个人可识别信息 | Theory 04, Lab 05 |
| RBAC | 基于角色的访问控制 | Theory 05, Lab 04 |
| Model Watermark | 在模型中嵌入可追踪的水印 | Theory 06, Lab 06 |
| Adversarial Attack | 通过微小扰动误导模型判断 | Theory 06, Lab 07 |
| TEE | 可信执行环境，保护推理过程 | Theory 06, Lab 08 |
| AI Governance | 企业级 AI 治理框架 | Theory 07, Lab 09-10 |

## 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装基础依赖
pip install torch transformers fastapi uvicorn
pip install nemoguardrails guardrails-ai
pip install presidio-analyzer presidio-anonymizer
pip install python-jose passlib bcrypt
pip install pydantic sqlalchemy redis
pip install pytest httpx

# 验证 GPU 可用
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

## 参考资源

- [OWASP LLM Top 10 (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI Risk Management Framework](https://www.nist.gov/artificial-intelligence/ai-risk-management-framework)
- [NeMo Guardrails Documentation](https://docs.nvidia.com/nemo/guardrails/)
- [Llama Guard Paper](https://arxiv.org/abs/2312.06674)
- [中国《生成式人工智能服务管理暂行办法》](http://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm)
- [EU AI Act](https://artificialintelligenceact.eu/)
