# JuiceFS 部署指南

## 环境要求

- Linux（推荐 Ubuntu 22.04+）
- Docker & Docker Compose
- 至少 500GB NVMe SSD（用于缓存）
- 网络畅通

## 第一步：部署 MinIO（对象存储后端）

```bash
# 创建数据目录
sudo mkdir -p /data/minio

# 部署 MinIO
docker run -d \
  --name minio \
  --restart always \
  -p 9000:9000 \
  -p 9001:9001 \
  -v /data/minio:/data \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin123 \
  minio/minio server /data --console-address ":9001"

# 验证
curl http://localhost:9000/minio/health/live
# 应返回 HTTP 200

# 安装 mc 客户端
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc && sudo mv mc /usr/local/bin/

# 配置 mc
mc alias set local http://localhost:9000 minioadmin minioadmin123

# 创建 bucket
mc mb local/jfs-data
mc mb local/jfs-checkpoint
```

## 第二步：部署 Redis（元数据引擎）

```bash
# 部署 Redis（带持久化）
docker run -d \
  --name redis-jfs \
  --restart always \
  -p 6379:6379 \
  -v /data/redis:/data \
  redis:7-alpine redis-server \
    --appendonly yes \
    --appendfsync everysec \
    --maxmemory 4gb \
    --maxmemory-policy noeviction

# 验证
redis-cli ping
# 应返回 PONG
```

## 第三步：安装 JuiceFS

```bash
# 一键安装
curl -sSL https://d.juicefs.com/install | sh -

# 验证
juicefs version
```

## 第四步：格式化 JuiceFS 卷

```bash
# 创建 AI 训练用的卷
juicefs format \
  --storage minio \
  --bucket http://localhost:9000/jfs-data \
  --access-key minioadmin \
  --secret-key minioadmin123 \
  --block-size 4096 \
  --compress none \
  redis://localhost:6379/1 \
  ai-training

# 创建 Checkpoint 专用卷
juicefs format \
  --storage minio \
  --bucket http://localhost:9000/jfs-checkpoint \
  --access-key minioadmin \
  --secret-key minioadmin123 \
  --block-size 4096 \
  redis://localhost:6379/2 \
  ai-checkpoint
```

## 第五步：挂载 JuiceFS

```bash
# 创建缓存目录
sudo mkdir -p /nvme/jfs-cache
sudo mkdir -p /mnt/jfs
sudo mkdir -p /mnt/jfs-ckpt

# 挂载训练数据卷（优化读性能）
juicefs mount \
  redis://localhost:6379/1 \
  /mnt/jfs \
  --cache-dir /nvme/jfs-cache/data \
  --cache-size 300000 \
  --prefetch 3 \
  --buffer-size 2048 \
  --max-uploads 30 \
  --metacache 3 \
  --entry-cache 3 \
  --attr-cache 3 \
  --metrics localhost:9567 \
  -d

# 挂载 Checkpoint 卷（优化写性能）
juicefs mount \
  redis://localhost:6379/2 \
  /mnt/jfs-ckpt \
  --cache-dir /nvme/jfs-cache/ckpt \
  --cache-size 100000 \
  --writeback \
  --buffer-size 4096 \
  --max-uploads 50 \
  --metrics localhost:9568 \
  -d

# 验证挂载
df -h /mnt/jfs
df -h /mnt/jfs-ckpt
```

## 第六步：验证和测试

```bash
# 基本读写测试
echo "hello juicefs" > /mnt/jfs/test.txt
cat /mnt/jfs/test.txt

# 性能测试
juicefs bench /mnt/jfs

# 查看实时统计
juicefs stats /mnt/jfs
```

## 第七步：配置开机自动挂载

```bash
# 创建 systemd service
sudo tee /etc/systemd/system/juicefs-data.service << 'EOF'
[Unit]
Description=JuiceFS Data Mount
After=docker.service redis.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/usr/local/bin/juicefs mount \
  redis://localhost:6379/1 \
  /mnt/jfs \
  --cache-dir /nvme/jfs-cache/data \
  --cache-size 300000 \
  --prefetch 3 \
  --buffer-size 2048 \
  --max-uploads 30 \
  -f
ExecStop=/usr/local/bin/juicefs umount /mnt/jfs
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable juicefs-data
```

## 运维命令速查

```bash
# 查看卷信息
juicefs info /mnt/jfs

# 查看缓存使用情况
juicefs stats /mnt/jfs | grep cache

# 预热模型到缓存
juicefs warmup -p 16 /mnt/jfs/models/

# 清理缓存
juicefs gc redis://localhost:6379/1

# IO 追踪（调试慢操作）
juicefs profile /mnt/jfs --interval 1

# 卸载
juicefs umount /mnt/jfs
```

## 常见问题

### Q: 缓存满了怎么办？
A: JuiceFS 会自动 LRU 淘汰。可通过 `--free-space-ratio 0.1` 预留 10% 空间。

### Q: Redis 挂了数据会丢吗？
A: 数据在对象存储中不会丢。但元数据需要 Redis 恢复后才能访问。建议配置 Redis 主从。

### Q: 写入性能不如预期？
A: 检查 `--writeback` 是否开启，`--buffer-size` 是否足够大，`--max-uploads` 是否足够。
