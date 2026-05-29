# 模型资产平台 Docker 镜像
# 多阶段构建：减小最终镜像体积

# ──────── Stage 1: 依赖安装 ────────
FROM python:3.11-slim AS builder

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ──────── Stage 2: 运行时 ────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# 从 builder 拷贝已安装的依赖
COPY --from=builder /install /usr/local

# 复制应用代码
COPY src/ ./src/
COPY tests/ ./tests/

# 创建非 root 用户
RUN useradd -m -r appuser && \
    mkdir -p /cache /data && \
    chown -R appuser:appuser /app /cache /data

USER appuser

# 环境变量默认值
ENV STORAGE_TYPE=local \
    STORAGE_ROOT=/data/models \
    CACHE_DIR=/cache \
    CACHE_MAX_GB=50 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# 启动命令：uvicorn + 多 worker
CMD ["python", "-m", "uvicorn", "src.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--timeout-keep-alive", "65"]
