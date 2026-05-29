"""
Lab 03: pgvector 向量数据库实战
使用 PostgreSQL + pgvector 扩展进行向量搜索
"""
import time
import numpy as np
import psycopg2
from psycopg2.extras import execute_values


# =============================================================================
# 连接 & 初始化
# =============================================================================

def setup_pgvector(host="localhost", port=5432, dbname="postgres",
                   user="postgres", password="postgres", dim=128):
    """连接 PostgreSQL 并启用 pgvector"""
    conn = psycopg2.connect(
        host=host, port=port, dbname=dbname,
        user=user, password=password,
    )
    conn.autocommit = True
    cur = conn.cursor()

    # 启用 pgvector 扩展
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 创建表
    cur.execute("DROP TABLE IF EXISTS doc_chunks;")
    cur.execute(f"""
        CREATE TABLE doc_chunks (
            id SERIAL PRIMARY KEY,
            doc_id VARCHAR(128),
            content TEXT,
            category VARCHAR(64),
            importance INTEGER,
            embedding vector({dim})
        );
    """)

    print(f"✓ pgvector 初始化完成，维度: {dim}")
    return conn


# =============================================================================
# 插入数据
# =============================================================================

def insert_data(conn, num_docs=10000, dim=128):
    """批量插入数据"""
    cur = conn.cursor()
    categories = ["技术", "业务", "HR", "财务", "产品"]

    batch_size = 1000
    total = 0

    for batch_start in range(0, num_docs, batch_size):
        batch_end = min(batch_start + batch_size, num_docs)
        values = []

        for i in range(batch_start, batch_end):
            vec = np.random.rand(dim).tolist()
            vec_str = f"[{','.join(str(v) for v in vec)}]"
            values.append((
                f"doc_{i}",
                f"这是第 {i} 个文档块的内容...",
                categories[i % len(categories)],
                np.random.randint(1, 11),
                vec_str,
            ))

        execute_values(
            cur,
            "INSERT INTO doc_chunks (doc_id, content, category, importance, embedding) VALUES %s",
            values,
        )
        total += len(values)

    conn.commit()
    print(f"✓ 插入 {total} 条记录")


# =============================================================================
# 创建索引
# =============================================================================

def create_index(conn, index_type="hnsw"):
    """创建向量索引"""
    cur = conn.cursor()

    # 删除旧索引
    cur.execute("DROP INDEX IF EXISTS doc_chunks_embedding_idx;")

    start_time = time.time()

    if index_type == "hnsw":
        cur.execute("""
            CREATE INDEX doc_chunks_embedding_idx
            ON doc_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 256);
        """)
    elif index_type == "ivfflat":
        # IVFFlat 需要先有数据
        cur.execute("""
            CREATE INDEX doc_chunks_embedding_idx
            ON doc_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)

    conn.commit()
    build_time = time.time() - start_time
    print(f"✓ 创建 {index_type} 索引，耗时: {build_time:.2f}s")


# =============================================================================
# 向量搜索
# =============================================================================

def search(conn, query_vector: list, top_k=5, category_filter=None):
    """
    向量搜索 - 使用 SQL！
    pgvector 的优势：可以无缝结合 SQL 过滤
    """
    cur = conn.cursor()
    vec_str = f"[{','.join(str(v) for v in query_vector)}]"

    if category_filter:
        # 带过滤的搜索
        cur.execute(f"""
            SELECT id, doc_id, category, importance,
                   1 - (embedding <=> %s::vector) as similarity
            FROM doc_chunks
            WHERE category = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vec_str, category_filter, vec_str, top_k))
    else:
        # 基本搜索
        cur.execute(f"""
            SELECT id, doc_id, category, importance,
                   1 - (embedding <=> %s::vector) as similarity
            FROM doc_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vec_str, vec_str, top_k))

    results = cur.fetchall()

    filter_info = f", filter={category_filter}" if category_filter else ""
    print(f"\npgvector 搜索 Top-{top_k}{filter_info}:")
    for row in results:
        print(f"  ID={row[0]}, Doc={row[1]}, Cat={row[2]}, "
              f"Importance={row[3]}, Similarity={row[4]:.4f}")

    return results


def advanced_sql_search(conn, query_vector: list):
    """
    高级 SQL 搜索：展示 pgvector + SQL 的强大组合
    """
    cur = conn.cursor()
    vec_str = f"[{','.join(str(v) for v in query_vector)}]"

    # 1. 聚合查询：每个分类中最相似的文档
    cur.execute(f"""
        SELECT DISTINCT ON (category)
            category, doc_id,
            1 - (embedding <=> %s::vector) as similarity
        FROM doc_chunks
        WHERE importance >= 5
        ORDER BY category, embedding <=> %s::vector
        LIMIT 5;
    """, (vec_str, vec_str))

    print("\n高级查询：每个分类最相似的文档 (重要性>=5):")
    for row in cur.fetchall():
        print(f"  Category={row[0]}, Doc={row[1]}, Similarity={row[2]:.4f}")

    # 2. 统计查询：相似度分布
    cur.execute(f"""
        SELECT
            CASE
                WHEN 1 - (embedding <=> %s::vector) > 0.9 THEN '0.9-1.0'
                WHEN 1 - (embedding <=> %s::vector) > 0.8 THEN '0.8-0.9'
                WHEN 1 - (embedding <=> %s::vector) > 0.7 THEN '0.7-0.8'
                ELSE '<0.7'
            END as similarity_range,
            COUNT(*) as count
        FROM doc_chunks
        GROUP BY similarity_range
        ORDER BY similarity_range DESC;
    """, (vec_str, vec_str, vec_str))

    print("\n相似度分布统计:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} 条")


# =============================================================================
# 主程序
# =============================================================================

def main():
    DIM = 128

    # 1. 初始化
    conn = setup_pgvector(dim=DIM)

    # 2. 插入数据
    insert_data(conn, num_docs=10000, dim=DIM)

    # 3. 创建索引
    create_index(conn, "hnsw")

    # 4. 搜索
    query = np.random.rand(DIM).tolist()
    search(conn, query, top_k=5)
    search(conn, query, top_k=5, category_filter="技术")

    # 5. 高级 SQL 搜索
    advanced_sql_search(conn, query)

    # 6. 清理
    conn.close()

    print(f"\n{'='*60}")
    print("pgvector 核心特性总结")
    print(f"{'='*60}")
    print("""
    1. SQL 原生接口，学习成本最低
    2. <=> 余弦距离, <-> L2 距离, <#> 内积
    3. 支持 HNSW 和 IVFFlat 索引
    4. 可与 JOIN/GROUP BY/DISTINCT ON 等 SQL 功能组合
    5. 事务支持（向量和业务数据一致性）
    6. 适合中小规模 + 已有 PG 基础设施
    """)


if __name__ == "__main__":
    main()
