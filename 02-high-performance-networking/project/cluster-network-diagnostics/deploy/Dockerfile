# 集群网络诊断Agent Dockerfile
# 基于Ubuntu 22.04，安装Mellanox OFED工具和诊断Agent

FROM ubuntu:22.04

LABEL maintainer="AI Infra Team"
LABEL description="Cluster Network Diagnostics Agent with OFED tools"

# 避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    openssh-client \
    wget \
    curl \
    gnupg2 \
    lsb-release \
    pciutils \
    net-tools \
    iproute2 \
    iputils-ping \
    ethtool \
    && rm -rf /var/lib/apt/lists/*

# 安装Mellanox OFED用户空间工具
# 注意：实际部署中应使用对应内核版本的OFED包
RUN wget -qO - https://linux.mellanox.com/public/keys/GPG-KEY-Mellanox.pub | apt-key add - && \
    echo "deb https://linux.mellanox.com/public/repo/mlnx_ofed/latest/ubuntu22.04/x86_64/ ./" \
    > /etc/apt/sources.list.d/mlnx_ofed.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ibverbs-utils \
    infiniband-diags \
    perftest \
    rdma-core \
    libibverbs-dev \
    librdmacm-dev \
    ibutils \
    mstflint \
    || echo "OFED packages not available in this environment, skipping..." && \
    rm -rf /var/lib/apt/lists/*

# 创建应用目录
WORKDIR /app

# 复制依赖文件并安装Python包
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ /app/src/
COPY config.yaml /app/config.yaml

# 创建数据目录
RUN mkdir -p /var/lib/cluster-diag /var/log/cluster-diag

# 创建非root运行用户（实际使用时可能需要root权限访问IB设备）
# RUN useradd -m -s /bin/bash diaguser

# 环境变量
ENV PYTHONPATH=/app
ENV DIAG_CONFIG_PATH=/app/config.yaml
ENV DIAG_LOG_LEVEL=INFO

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import src.discovery.rdma_device_scanner; print('OK')" || exit 1

# 入口点 - 运行诊断Agent
ENTRYPOINT ["python3", "-m", "src.main"]
CMD ["--config", "/app/config.yaml", "--mode", "daemon"]
