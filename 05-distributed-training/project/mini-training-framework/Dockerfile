FROM nvcr.io/nvidia/pytorch:24.01-py3

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# 项目代码
COPY . /workspace/mini-training-framework
WORKDIR /workspace/mini-training-framework

# 环境变量
ENV NCCL_DEBUG=WARN
ENV NCCL_IB_DISABLE=0
ENV CUDA_DEVICE_MAX_CONNECTIONS=1

# 默认入口
CMD ["torchrun", "--nproc_per_node=8", "-m", "src.parallel.hybrid_parallel", \
     "--config", "configs/8gpu_tp4_pp2.yaml"]
