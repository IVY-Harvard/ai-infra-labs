# Lab 03: 向量数据库实战

## 目标
动手操作 Milvus、Qdrant、pgvector 三种向量数据库，并进行性能对比。

## 前置准备
```bash
# 启动数据库
docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
docker run -d --name pgvector -p 5432:5432 -e POSTGRES_PASSWORD=postgres ankane/pgvector:latest

pip install pymilvus qdrant-client psycopg2-binary pgvector
```

## 实验内容
1. `milvus_demo.py` — Milvus CRUD + 索引 + 搜索
2. `qdrant_demo.py` — Qdrant CRUD + 过滤 + 搜索
3. `pgvector_demo.py` — pgvector SQL 操作
4. `benchmark.py` — 三者性能对比（插入/查询/过滤）
