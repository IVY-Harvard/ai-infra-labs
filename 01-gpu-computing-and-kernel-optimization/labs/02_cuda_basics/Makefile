# Makefile for CUDA labs
# 使用方法: make / make clean

NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_90  # sm_90 for Hopper (H20/H100)
# 如果你的 GPU 不是 Hopper，修改 -arch:
# V100: sm_70, T4: sm_75, A100: sm_80, RTX 3090: sm_86

TARGET = vector_add

all: $(TARGET)

$(TARGET): vector_add.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

# 用 NSight Compute 做性能分析
profile: $(TARGET)
	ncu --set full ./$(TARGET)

.PHONY: all clean profile
