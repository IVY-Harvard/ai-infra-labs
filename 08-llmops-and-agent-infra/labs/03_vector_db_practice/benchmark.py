"""
Lab 03: 向量数据库性能对比 Benchmark
对比 Milvus / Qdrant / pgvector 的插入和查询性能
"""
import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class BenchmarkResult:
    db_name: str
    operation: str
    num_vectors: int
    dimension: int
    duration_ms: float
    throughput: float  # ops/sec
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None


class VectorDBBenchmark:
    """向量数据库统一 Benchmark 框架"""

    def __init__(self, dim: int = 128, num_vectors: int = 50000):
        self.dim = dim
        self.num_vectors = num_vectors
        self.results: list[BenchmarkResult] = []

    # ----- Milvus Benchmark -----

    def benchmark_milvus(self):
        """Milvus 性能测试"""
        print("\n" + "=" * 60)
        print("Benchmarking Milvus")
        print("=" * 60)

        try:
            from pymilvus import (
                connections, Collection, CollectionSchema,
                FieldSchema, DataType, utility,
            )

            connections.connect("default", host="localhost", port=19530)

            # 创建 Collection
            if utility.has_collection("benchmark"):
                Collection("benchmark").drop()

            fields = [
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema("category", DataType.VARCHAR, max_length=64),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=self.dim),
            ]
            collection = Collection("benchmark",
                                    CollectionSchema(fields))

            # 插入 Benchmark
            vectors = np.random.rand(self.num_vectors, self.dim).astype(np.float32)
            categories = [f"cat_{i % 10}" for i in range(self.num_vectors)]

            start = time.time()
            batch_size = 5000
            for i in range(0, self.num_vectors, batch_size):
                end = min(i + batch_size, self.num_vectors)
                collection.insert([
                    categories[i:end],
                    vectors[i:end].tolist(),
                ])
            collection.flush()
            insert_time = time.time() - start

            self.results.append(BenchmarkResult(
                db_name="Milvus", operation="insert",
                num_vectors=self.num_vectors, dimension=self.dim,
                duration_ms=insert_time * 1000,
                throughput=self.num_vectors / insert_time,
            ))
            print(f"  Insert: {insert_time:.2f}s, "
                  f"{self.num_vectors / insert_time:.0f} vec/s")

            # 创建索引
            collection.create_index("embedding", {
                "index_type": "HNSW", "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 256},
            })
            collection.load()

            # 查询 Benchmark
            query_vectors = np.random.rand(100, self.dim).tolist()
            latencies = []

            for qv in query_vectors:
                start = time.time()
                collection.search(
                    [qv], "embedding",
                    {"metric_type": "COSINE", "params": {"ef": 128}},
                    limit=10,
                )
                latencies.append((time.time() - start) * 1000)

            latencies.sort()
            self.results.append(BenchmarkResult(
                db_name="Milvus", operation="search",
                num_vectors=self.num_vectors, dimension=self.dim,
                duration_ms=sum(latencies),
                throughput=len(latencies) / (sum(latencies) / 1000),
                p50_latency_ms=latencies[49],
                p95_latency_ms=latencies[94],
                p99_latency_ms=latencies[98],
            ))
            print(f"  Search: P50={latencies[49]:.1f}ms, "
                  f"P95={latencies[94]:.1f}ms, P99={latencies[98]:.1f}ms")

            collection.release()

        except Exception as e:
            print(f"  Milvus 不可用: {e}")

    # ----- Qdrant Benchmark -----

    def benchmark_qdrant(self):
        """Qdrant 性能测试"""
        print("\n" + "=" * 60)
        print("Benchmarking Qdrant")
        print("=" * 60)

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance, VectorParams, PointStruct,
            )

            client = QdrantClient(host="localhost", port=6333)

            if client.collection_exists("benchmark"):
                client.delete_collection("benchmark")

            client.create_collection("benchmark", VectorParams(
                size=self.dim, distance=Distance.COSINE,
            ))

            # 插入 Benchmark
            vectors = np.random.rand(self.num_vectors, self.dim)

            start = time.time()
            batch_size = 5000
            for i in range(0, self.num_vectors, batch_size):
                end = min(i + batch_size, self.num_vectors)
                points = [
                    PointStruct(id=j, vector=vectors[j].tolist(),
                                payload={"category": f"cat_{j % 10}"})
                    for j in range(i, end)
                ]
                client.upsert("benchmark", points)
            insert_time = time.time() - start

            self.results.append(BenchmarkResult(
                db_name="Qdrant", operation="insert",
                num_vectors=self.num_vectors, dimension=self.dim,
                duration_ms=insert_time * 1000,
                throughput=self.num_vectors / insert_time,
            ))
            print(f"  Insert: {insert_time:.2f}s, "
                  f"{self.num_vectors / insert_time:.0f} vec/s")

            # 查询 Benchmark
            query_vectors = np.random.rand(100, self.dim).tolist()
            latencies = []

            for qv in query_vectors:
                start = time.time()
                client.search("benchmark", query_vector=qv, limit=10)
                latencies.append((time.time() - start) * 1000)

            latencies.sort()
            self.results.append(BenchmarkResult(
                db_name="Qdrant", operation="search",
                num_vectors=self.num_vectors, dimension=self.dim,
                duration_ms=sum(latencies),
                throughput=len(latencies) / (sum(latencies) / 1000),
                p50_latency_ms=latencies[49],
                p95_latency_ms=latencies[94],
                p99_latency_ms=latencies[98],
            ))
            print(f"  Search: P50={latencies[49]:.1f}ms, "
                  f"P95={latencies[94]:.1f}ms, P99={latencies[98]:.1f}ms")

        except Exception as e:
            print(f"  Qdrant 不可用: {e}")

    # ----- pgvector Benchmark -----

    def benchmark_pgvector(self):
        """pgvector 性能测试"""
        print("\n" + "=" * 60)
        print("Benchmarking pgvector")
        print("=" * 60)

        try:
            import psycopg2
            from psycopg2.extras import execute_values

            conn = psycopg2.connect(
                host="localhost", port=5432, dbname="postgres",
                user="postgres", password="postgres",
            )
            conn.autocommit = True
            cur = conn.cursor()

            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("DROP TABLE IF EXISTS benchmark;")
            cur.execute(f"""
                CREATE TABLE benchmark (
                    id SERIAL PRIMARY KEY,
                    category VARCHAR(64),
                    embedding vector({self.dim})
                );
            """)

            # 插入 Benchmark
            vectors = np.random.rand(self.num_vectors, self.dim)

            start = time.time()
            batch_size = 5000
            for i in range(0, self.num_vectors, batch_size):
                end = min(i + batch_size, self.num_vectors)
                values = []
                for j in range(i, end):
                    vec_str = f"[{','.join(str(v) for v in vectors[j])}]"
                    values.append((f"cat_{j % 10}", vec_str))
                execute_values(
                    cur,
                    "INSERT INTO benchmark (category, embedding) VALUES %s",
                    values,
                )
            conn.commit()
            insert_time = time.time() - start

            self.results.append(BenchmarkResult(
                db_name="pgvector", operation="insert",
                num_vectors=self.num_vectors, dimension=self.dim,
                duration_ms=insert_time * 1000,
                throughput=self.num_vectors / insert_time,
            ))
            print(f"  Insert: {insert_time:.2f}s, "
                  f"{self.num_vectors / insert_time:.0f} vec/s")

            # 创建索引
            cur.execute("""
                CREATE INDEX ON benchmark
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 256);
            """)

            # 查询 Benchmark
            query_vectors = np.random.rand(100, self.dim)
            latencies = []

            for qv in query_vectors:
                vec_str = f"[{','.join(str(v) for v in qv)}]"
                start = time.time()
                cur.execute(f"""
                    SELECT id FROM benchmark
                    ORDER BY embedding <=> %s::vector
                    LIMIT 10;
                """, (vec_str,))
                cur.fetchall()
                latencies.append((time.time() - start) * 1000)

            latencies.sort()
            self.results.append(BenchmarkResult(
                db_name="pgvector", operation="search",
                num_vectors=self.num_vectors, dimension=self.dim,
                duration_ms=sum(latencies),
                throughput=len(latencies) / (sum(latencies) / 1000),
                p50_latency_ms=latencies[49],
                p95_latency_ms=latencies[94],
                p99_latency_ms=latencies[98],
            ))
            print(f"  Search: P50={latencies[49]:.1f}ms, "
                  f"P95={latencies[94]:.1f}ms, P99={latencies[98]:.1f}ms")

            conn.close()

        except Exception as e:
            print(f"  pgvector 不可用: {e}")

    # ----- 汇总报告 -----

    def print_report(self):
        """打印对比报告"""
        print("\n" + "=" * 80)
        print(f"向量数据库性能对比报告 ({self.num_vectors} 向量, {self.dim} 维)")
        print("=" * 80)

        # 插入性能
        print(f"\n{'数据库':<12} {'插入耗时(s)':<14} {'吞吐(vec/s)':<14}")
        print("-" * 40)
        for r in self.results:
            if r.operation == "insert":
                print(f"{r.db_name:<12} {r.duration_ms/1000:<14.2f} {r.throughput:<14.0f}")

        # 查询性能
        print(f"\n{'数据库':<12} {'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} {'QPS':<10}")
        print("-" * 52)
        for r in self.results:
            if r.operation == "search":
                print(f"{r.db_name:<12} {r.p50_latency_ms:<10.1f} "
                      f"{r.p95_latency_ms:<10.1f} {r.p99_latency_ms:<10.1f} "
                      f"{r.throughput:<10.0f}")

        # 保存结果
        with open("benchmark_results.json", "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print("\n✓ 结果已保存到 benchmark_results.json")


def main():
    bench = VectorDBBenchmark(dim=128, num_vectors=50000)

    bench.benchmark_milvus()
    bench.benchmark_qdrant()
    bench.benchmark_pgvector()

    bench.print_report()


if __name__ == "__main__":
    main()
