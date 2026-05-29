FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 默认入口
ENTRYPOINT ["python3", "-m", "src.cli"]
CMD ["benchmark", "--all"]
