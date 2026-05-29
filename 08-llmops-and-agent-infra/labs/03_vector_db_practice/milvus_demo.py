"""
Lab 03: Milvus 向量数据库实战
涵盖：连接、建表、插入、索引、搜索、过滤
"""
import time
import numpy as np
from pymilvus import (
    connections, utility,
    Collection, CollectionSchema, FieldSchema,
    DataType,
)


# =============================================================================
# 连接
# =============================================================================

def connect_milvus(host: str = "localhost", port: int = 19530):
    """连接 Milvus"""
    connections.connect("default", host=host, port=port)
    print(f"✓ 已连接 Milvus {host}:{port}")
    print(f"  已有 collections: {utility.list_collections()}")


# =============================================================================
# Schema 定义 & 创建 Collection
# =============================================================================

def create_collection(name: str = "doc_chunks", dim: int = 1024):
    """创建 Collection（类比关系数据库的表）"""
    # 删除已存在的同名 collection
    if utility.has_collection(name):
        Collection(name).drop()
        print(f"  已删除旧 collection: {name}")

    # 定义 Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64,
                    is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="Document chunks with embeddings")

    collection = Collection(name, schema)
    print(f"✓ 创建 collection: {name}, 维度: {dim}")
    return collection


# =============================================================================
# 插入数据
# =============================================================================

def insert_data(collection: Collection, num_docs: int = 100,
                dim: int = 1024) -> int:
    """批量插入测试数据"""
    batch_size = 1000
    total_inserted = 0

    for batch_start in range(0, num_docs, batch_size):
        batch_end = min(batch_start + batch_size, num_docs)
        batch_count = batch_end - batch_start

        data = [
            [f"doc_{i}" for i in range(batch_start, batch_end)],           # doc_id
            [i % 10 for i in range(batch_start, batch_end)],               # chunk_index
            [f"这是第 {i} 个文档块的内容..." for i in range(batch_start, batch_end)],  # content
            [f"source_{i % 5}.pdf" for i in range(batch_start, batch_end)],  # source
            np.random.rand(batch_count, dim).tolist(),                       # embedding
        ]

        result = collection.insert(data)
        total_inserted += batch_count

    collection.flush()
    print(f"✓ 插入 {total_inserted} 条记录")
    return total_inserted


# =============================================================================
# 创建索引
# =============================================================================

def create_index(collection: Collection, index_type: str = "HNSW"):
    """创建向量索引"""
    index_configs = {
        "HNSW": {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 256},
        },
        "IVF_FLAT": {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        },
        "IVF_PQ": {
            "index_type": "IVF_PQ",
            "metric_type": "L2",
            "params": {"nlist": 128, "m": 16, "nbits": 8},
        },
    }

    config = index_configs[index_type]

    start_time = time.time()
    collection.create_index("embedding", config)
    build_time = time.time() - start_time

    print(f"✓ 创建 {index_type} 索引，耗时: {build_time:.2f}s")
    return build_time


# =============================================================================
# 向量搜索
# =============================================================================

def search(collection: Collection, query_vector: list,
           top_k: int = 5, expr: str = None):
    """
    向量搜索
    expr: 标量过滤表达式，如 'source == "source_1.pdf"'
    """
    collection.load()

    search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

    start_time = time.time()
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["doc_id", "content", "source"],
    )
    search_time = (time.time() - start_time) * 1000

    print(f"\n搜索结果 (Top-{top_k}, 耗时: {search_time:.1f}ms):")
    if expr:
        print(f"  过滤条件: {expr}")

    for hits in results:
        for hit in hits:
            print(f"  ID={hit.id}, Score={hit.score:.4f}, "
                  f"Doc={hit.entity.get('doc_id')}, "
                  f"Source={hit.entity.get('source')}")

    return results, search_time


# =============================================================================
# 主程序
# =============================================================================

def main():
    DIM = 128  # 使用较小维度加速演示

    # 1. 连接
    connect_milvus()

    # 2. 创建 Collection
    collection = create_collection("demo_collection", dim=DIM)

    # 3. 插入数据
    insert_data(collection, num_docs=10000, dim=DIM)

    # 4. 创建索引
    for idx_type in ["HNSW", "IVF_FLAT"]:
        # 删除旧索引
        collection.release()
        collection.drop_index()
        create_index(collection, idx_type)

        # 5. 搜索
        query = np.random.rand(DIM).tolist()

        print(f"\n--- {idx_type} 索引搜索 ---")
        search(collection, query, top_k=5)

        # 6. 带过滤的搜索
        print(f"\n--- {idx_type} 带过滤搜索 ---")
        search(collection, query, top_k=5, expr='source == "source_1.pdf"')

    # 7. 清理
    collection.release()
    print("\n✓ 演示完成")

    # 打印总结
    print(f"\n{'='*60}")
    print("Milvus 核心操作总结")
    print(f"{'='*60}")
    print("""
    1. Collection = 表，Field = 列
    2. 必须创建索引才能高效搜索
    3. search 前需要 load() 到内存
    4. expr 支持标量过滤（类似 SQL WHERE）
    5. 不同索引类型适用不同场景
       HNSW: 高精度，内存大
       IVF_FLAT: 平衡选择
       IVF_PQ: 内存节省，精度稍低
    """)


if __name__ == "__main__":
    main()
