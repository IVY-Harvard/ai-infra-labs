# Lab 10：数据飞轮

## 实验目标

搭建完整的数据飞轮闭环：模型上线 → 反馈收集 → 标注 → 训练 → 更新模型。

## 实验内容

### 实验 1：反馈收集器
收集用户对模型输出的反馈（点赞/点踩/编辑/重新生成）。

### 实验 2：标注流水线
实现分层标注（自动标注 → 众包 → 专家）。

### 实验 3：飞轮编排器
编排整个飞轮的自动化流程。

## 运行方式

```bash
pip install fastapi uvicorn pydantic

# 反馈收集器
python feedback_collector.py --port 8001

# 标注流水线
python annotation_pipeline.py --input feedback_data.jsonl --output annotated.jsonl

# 飞轮编排器
python flywheel_orchestrator.py --config flywheel_config.json
```

## 文件列表

- `feedback_collector.py` — 用户反馈收集服务
- `annotation_pipeline.py` — 分层标注流水线
- `flywheel_orchestrator.py` — 飞轮编排器
