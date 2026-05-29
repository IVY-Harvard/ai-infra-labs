# Alluxio 单节点部署指南

## 环境要求

- Java 11+
- Docker（用于部署依赖组件）
- 至少 64GB 内存（Alluxio Worker 需要内存缓存）
- NVMe SSD（用于 SSD 层缓存）

## 第一步：下载 Alluxio

```bash
# 下载 Alluxio 2.9
wget https://downloads.alluxio.io/downloads/files/2.9.5/alluxio-2.9.5-bin.tar.gz
tar xzf alluxio-2.9.5-bin.tar.gz
cd alluxio-2.9.5

export ALLUXIO_HOME=$(pwd)
```

## 第二步：配置 Alluxio

```bash
# 创建配置文件
cp conf/alluxio-site.properties.template conf/alluxio-site.properties

cat > conf/alluxio-site.properties << 'EOF'
# --- Master 配置 ---
alluxio.master.hostname=localhost
alluxio.master.mount.table.root.ufs=s3://jfs-data/

# --- S3 后端配置（MinIO）---
alluxio.underfs.s3.endpoint=http://localhost:9000
alluxio.underfs.s3.disable.dns.buckets=true
aws.accessKeyId=minioadmin
aws.secretKey=minioadmin123

# --- Worker 缓存配置 ---
alluxio.worker.tieredstore.levels=2

# 第一层：内存
alluxio.worker.tieredstore.level0.alias=MEM
alluxio.worker.tieredstore.level0.dirs.path=/dev/shm/alluxio
alluxio.worker.tieredstore.level0.dirs.quota=32GB
alluxio.worker.tieredstore.level0.watermark.high.ratio=0.95
alluxio.worker.tieredstore.level0.watermark.low.ratio=0.7

# 第二层：SSD
alluxio.worker.tieredstore.level1.alias=SSD
alluxio.worker.tieredstore.level1.dirs.path=/nvme/alluxio
alluxio.worker.tieredstore.level1.dirs.quota=300GB
alluxio.worker.tieredstore.level1.watermark.high.ratio=0.95
alluxio.worker.tieredstore.level1.watermark.low.ratio=0.7

# --- 读写策略 ---
alluxio.user.file.readtype.default=CACHE
alluxio.user.file.writetype.default=ASYNC_THROUGH

# --- AI 训练优化 ---
alluxio.user.streaming.reader.chunk.size.bytes=8MB
alluxio.user.local.reader.chunk.size.bytes=8MB
alluxio.user.file.passive.cache.enabled=true
alluxio.user.file.replication.min=1

# --- 性能调优 ---
alluxio.user.network.reader.buffer.size.bytes=8MB
alluxio.user.network.writer.buffer.size.bytes=8MB
EOF
```

## 第三步：启动 Alluxio

```bash
# 格式化（首次运行）
$ALLUXIO_HOME/bin/alluxio format

# 启动 Master + Worker
$ALLUXIO_HOME/bin/alluxio-start.sh local SudoMount

# 验证
$ALLUXIO_HOME/bin/alluxio fsadmin report

# Web UI
# 打开浏览器访问 http://localhost:19999
```

## 第四步：挂载底层存储

```bash
# 挂载 MinIO bucket 到 Alluxio 命名空间
$ALLUXIO_HOME/bin/alluxio fs mount \
  /training-data \
  s3://jfs-data/training-data/

# 验证
$ALLUXIO_HOME/bin/alluxio fs ls /training-data/
```

## 第五步：POSIX 访问（FUSE）

```bash
# 通过 FUSE 挂载为本地目录
sudo mkdir -p /mnt/alluxio
$ALLUXIO_HOME/integration/fuse/bin/alluxio-fuse mount \
  /mnt/alluxio /

# 验证
ls /mnt/alluxio/training-data/
```

## 运维命令

```bash
# 查看缓存状态
$ALLUXIO_HOME/bin/alluxio fsadmin report capacity

# 手动预加载数据到缓存
$ALLUXIO_HOME/bin/alluxio fs load /training-data/ --local

# 释放缓存
$ALLUXIO_HOME/bin/alluxio fs free /training-data/old-data/

# 查看指定文件缓存位置
$ALLUXIO_HOME/bin/alluxio fs location /training-data/model.bin
```
