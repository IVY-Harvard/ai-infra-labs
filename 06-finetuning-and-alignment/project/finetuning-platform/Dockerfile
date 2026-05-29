FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 默认端口
EXPOSE 8080

# 启动 API 服务
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8080"]
