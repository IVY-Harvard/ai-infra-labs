"""
Lab 03: Qdrant 向量数据库实战
涵盖：连接、创建 Collection、插入、搜索、高级过滤
"""
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    SearchParams, HnswConfigDiff,
)


# =============================================================================
# 连接 & 创建 Collection
# =============================================================================

def setup_qdrant(host: str = "localhost", port: int = 6333,
                 dim: int = 128) -> QdrantClient:
    """连接 Qdrant 并创建 Collection"""
    client = QdrantClient(host=host, port=port)
    print(f"✓ 已连接 Qdrant {host}:{port}")

    collection_name = "doc_chunks"

    # 重建 collection
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=256),
        ),
    )
    print(f"✓ 创建 collection: {collection_name}")
    return client


# =============================================================================
# 插入数据
# =============================================================================

def insert_data(client: QdrantClient, num_docs: int = 10000,
                dim: int = 128):
    """批量插入数据（带丰富的 Payload）"""
    collection_name = "doc_chunks"
    batch_size = 1000

    categories = ["技术", "业务", "HR", "财务", "产品"]
    departments = ["工程部", "市场部", "人力资源", "财务部", "产品部"]

    for batch_start in range(0, num_docs, batch_size):
        batch_end = min(batch_start + batch_size, num_docs)
        points = []

        for i in range(batch_start, batch_end):
            points.append(PointStruct(
                id=i,
                vector=np.random.rand(dim).tolist(),
                payload={
                    "doc_id": f"doc_{i}",
                    "content": f"这是第 {i} 个文档块的内容...",
                    "category": categories[i % len(categories)],
                    "department": departments[i % len(departments)],
                    "importance": np.random.randint(1, 11),
                    "word_count": np.random.randint(50, 500),
                    "timestamp": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                },
            ))

        client.upsert(collection_name=collection_name, points=points)

    print(f"✓ 插入 {num_docs} 条记录")


# =============================================================================
# 向量搜索
# =============================================================================

def basic_search(client: QdrantClient, query_vector: list,
                 top_k: int = 5):
    """基本向量搜索"""
    results = client.search(
        collection_name="doc_chunks",
        query_vector=query_vector,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128),
    )

    print(f"\n基本搜索 Top-{top_k}:")
    for r in results:
        print(f"  ID={r.id}, Score={r.score:.4f}, "
              f"Category={r.payload.get('category')}")
    return results


def filtered_search(client: QdrantClient, query_vector: list,
                    top_k: int = 5):
    """带过滤条件的搜索 — Qdrant 的强项"""
    # 过滤条件：技术文档 + 重要性 >= 7
    results = client.search(
        collection_name="doc_chunks",
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="category", match=MatchValue(value="技术")),
                FieldCondition(key="importance", range=Range(gte=7)),
            ]
        ),
        limit=top_k,
    )

    print(f"\n过滤搜索 (技术 + 重要性>=7) Top-{top_k}:")
    for r in results:
        print(f"  ID={r.id}, Score={r.score:.4f}, "
              f"Category={r.payload.get('category')}, "
              f"Importance={r.payload.get('importance')}")
    return results


def complex_filtered_search(client: QdrantClient, query_vector: list):
    """复杂过滤：组合 must / should / must_not"""
    results = client.search(
        collection_name="doc_chunks",
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="word_count", range=Range(gte=100, lte=300)),
            ],
            should=[
                FieldCondition(key="category", match=MatchValue(value="技术")),
                FieldCondition(key="category", match=MatchValue(value="产品")),
            ],
            must_not=[
                FieldCondition(key="department", match=MatchValue(value="HR")),
            ],
        ),
        limit=5,
    )

    print(f"\n复杂过滤搜索 (100<=字数<=300, 技术或产品, 非HR):")
    for r in results:
        p = r.payload
        print(f"  ID={r.id}, Score={r.score:.4f}, "
              f"Cat={p.get('category')}, Dept={p.get('department')}, "
              f"Words={p.get('word_count')}")
    return results


# =============================================================================
# 主程序
# =============================================================================

def main():
    DIM = 128

    # 1. 初始化
    client = setup_qdrant(dim=DIM)

    # 2. 插入数据
    insert_data(client, num_docs=10000, dim=DIM)

    # 3. 搜索
    query = np.random.rand(DIM).tolist()
    basic_search(client, query)
    filtered_search(client, query)
    complex_filtered_search(client, query)

    # 4. Collection 信息
    info = client.get_collection("doc_chunks")
    print(f"\nCollection 信息:")
    print(f"  向量数: {info.points_count}")
    print(f"  索引状态: {info.status}")

    print(f"\n{'='*60}")
    print("Qdrant 核心特性总结")
    print(f"{'='*60}")
    print("""
    1. Payload = 元数据，支持丰富的过滤
    2. Filter 支持 must/should/must_not 组合
    3. 支持 Range、Match、Geo 等多种条件
    4. 无需显式 load，自动管理内存
    5. 内置量化支持，可减少 4x 内存
    6. Rust 实现，单机性能优异
    """)


if __name__ == "__main__":
    main()
