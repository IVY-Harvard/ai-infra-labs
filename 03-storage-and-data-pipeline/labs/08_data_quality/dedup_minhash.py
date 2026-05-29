"""
MinHash 近似去重

使用 MinHash + LSH（局部敏感哈希）实现大规模文本近似去重。
适合处理 GB-TB 级别的文本数据集。

用法：
    python dedup_minhash.py --input-dir /path/to/texts --threshold 0.8
    python dedup_minhash.py --generate-demo --input-dir /tmp/dedup_demo
"""

import os
import time
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path
from typing import List, Set, Tuple, Dict
from collections import defaultdict


class MinHashDedup:
    """MinHash + LSH 近似去重

    参数选择指南：
    - num_perm=128, bands=16, rows=8: Jaccard > 0.5 的近似对
    - num_perm=128, bands=32, rows=4: 更激进（检出率高但误报多）
    - num_perm=256, bands=16, rows=16: 更保守（只去很相似的）
    """

    def __init__(self, num_perm: int = 128, bands: int = 16,
                 rows: int = 8, ngram_size: int = 5,
                 threshold: float = 0.8):
        assert bands * rows == num_perm
        self.num_perm = num_perm
        self.bands = bands
        self.rows = rows
        self.ngram_size = ngram_size
        self.threshold = threshold

        # 哈希函数参数
        rng = np.random.RandomState(42)
        self.hash_a = rng.randint(1, 2**31 - 1, size=num_perm,
                                   dtype=np.int64)
        self.hash_b = rng.randint(0, 2**31 - 1, size=num_perm,
                                   dtype=np.int64)
        self.prime = (1 << 31) - 1

        # LSH 桶
        self.buckets: List[Dict[str, List[str]]] = [
            {} for _ in range(bands)
        ]
        self.signatures: Dict[str, np.ndarray] = {}

    def get_ngrams(self, text: str) -> Set[str]:
        """提取 word-level n-gram"""
        words = text.lower().split()
        if len(words) < self.ngram_size:
            return {" ".join(words)} if words else set()
        return {
            " ".join(words[i:i + self.ngram_size])
            for i in range(len(words) - self.ngram_size + 1)
        }

    def compute_minhash(self, ngrams: Set[str]) -> np.ndarray:
        """计算 MinHash 签名"""
        signature = np.full(self.num_perm, np.iinfo(np.int64).max,
                            dtype=np.int64)

        for ngram in ngrams:
            h = int(hashlib.sha256(ngram.encode()).hexdigest()[:16], 16)
            hashes = (self.hash_a * h + self.hash_b) % self.prime
            signature = np.minimum(signature, hashes)

        return signature

    def insert(self, doc_id: str, text: str):
        """插入文档"""
        ngrams = self.get_ngrams(text)
        if not ngrams:
            return

        sig = self.compute_minhash(ngrams)
        self.signatures[doc_id] = sig

        # 插入 LSH 桶
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band_hash = hashlib.md5(
                sig[start:end].tobytes()
            ).hexdigest()

            if band_hash not in self.buckets[band_idx]:
                self.buckets[band_idx][band_hash] = []
            self.buckets[band_idx][band_hash].append(doc_id)

    def find_duplicates(self, doc_id: str) -> Set[str]:
        """查找与给定文档相似的候选"""
        if doc_id not in self.signatures:
            return set()

        sig = self.signatures[doc_id]
        candidates = set()

        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band_hash = hashlib.md5(
                sig[start:end].tobytes()
            ).hexdigest()

            if band_hash in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][band_hash])

        candidates.discard(doc_id)
        return candidates

    def jaccard_similarity(self, doc_id_a: str, doc_id_b: str) -> float:
        """用 MinHash 签名估计 Jaccard 相似度"""
        sig_a = self.signatures.get(doc_id_a)
        sig_b = self.signatures.get(doc_id_b)
        if sig_a is None or sig_b is None:
            return 0.0
        return float(np.mean(sig_a == sig_b))

    def deduplicate(self, documents: List[Tuple[str, str]]) -> Dict:
        """执行去重

        Args:
            documents: [(doc_id, text), ...]

        Returns:
            {"kept": [...], "removed": [...], "clusters": [...]}
        """
        print(f"开始去重: {len(documents)} 篇文档")
        print(f"  参数: num_perm={self.num_perm}, bands={self.bands}, "
              f"rows={self.rows}, threshold={self.threshold}")

        # 阶段 1: 计算签名并插入 LSH
        t0 = time.perf_counter()
        for doc_id, text in documents:
            self.insert(doc_id, text)
        insert_time = time.perf_counter() - t0
        print(f"  签名计算+LSH插入: {insert_time:.2f}s")

        # 阶段 2: 查找重复对
        t0 = time.perf_counter()
        removed = set()
        duplicate_pairs = []

        for doc_id, _ in documents:
            if doc_id in removed:
                continue

            candidates = self.find_duplicates(doc_id)
            for candidate_id in candidates:
                if candidate_id in removed or candidate_id == doc_id:
                    continue

                sim = self.jaccard_similarity(doc_id, candidate_id)
                if sim >= self.threshold:
                    removed.add(candidate_id)
                    duplicate_pairs.append((doc_id, candidate_id, sim))

        query_time = time.perf_counter() - t0
        print(f"  重复检测: {query_time:.2f}s")

        kept = [doc_id for doc_id, _ in documents if doc_id not in removed]

        print(f"\n  结果:")
        print(f"    总文档: {len(documents)}")
        print(f"    保留: {len(kept)} ({len(kept)/len(documents)*100:.1f}%)")
        print(f"    去除: {len(removed)} ({len(removed)/len(documents)*100:.1f}%)")
        print(f"    重复对: {len(duplicate_pairs)}")

        if duplicate_pairs[:5]:
            print(f"\n  前 5 个重复对（相似度）:")
            for a, b, sim in sorted(duplicate_pairs, key=lambda x: -x[2])[:5]:
                print(f"    {a} <-> {b}: {sim:.3f}")

        return {
            "kept": kept,
            "removed": list(removed),
            "duplicate_pairs": duplicate_pairs,
            "stats": {
                "total": len(documents),
                "kept": len(kept),
                "removed": len(removed),
                "dedup_ratio": len(removed) / len(documents),
            }
        }


def generate_demo_data(output_dir: str, num_docs: int = 1000,
                       dup_ratio: float = 0.2):
    """生成包含重复的演示数据"""
    os.makedirs(output_dir, exist_ok=True)

    templates = [
        "The quick brown fox jumps over the lazy dog in the park",
        "Machine learning models require large amounts of training data",
        "Deep neural networks have revolutionized natural language processing",
        "Distributed storage systems enable scalable data management",
        "GPU acceleration has made training large models feasible",
    ]

    documents = []
    for i in range(num_docs):
        if i > 0 and np.random.random() < dup_ratio:
            # 创建近似重复（稍微修改原文）
            source_idx = np.random.randint(0, len(documents))
            text = documents[source_idx][1]
            words = text.split()
            # 替换 1-3 个词
            for _ in range(np.random.randint(1, 4)):
                if words:
                    idx = np.random.randint(0, len(words))
                    words[idx] = f"word{np.random.randint(0, 100)}"
            text = " ".join(words)
        else:
            # 原创文档
            template = templates[np.random.randint(0, len(templates))]
            extra = " ".join([f"token{np.random.randint(0, 500)}"
                            for _ in range(30)])
            text = f"{template} {extra}"

        documents.append((f"doc_{i:06d}", text))

    # 保存
    filepath = os.path.join(output_dir, "documents.jsonl")
    with open(filepath, "w") as f:
        for doc_id, text in documents:
            f.write(json.dumps({"id": doc_id, "text": text}) + "\n")

    print(f"生成 {num_docs} 篇文档 (约 {dup_ratio*100:.0f}% 近似重复)")
    return documents


def main():
    parser = argparse.ArgumentParser(description="MinHash 近似去重")
    parser.add_argument("--input-dir", type=str, default="/tmp/dedup_demo")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--generate-demo", action="store_true")
    args = parser.parse_args()

    if args.generate_demo:
        documents = generate_demo_data(args.input_dir)
    else:
        # 从文件加载
        filepath = os.path.join(args.input_dir, "documents.jsonl")
        documents = []
        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                documents.append((data["id"], data["text"]))

    dedup = MinHashDedup(
        num_perm=args.num_perm,
        ngram_size=args.ngram_size,
        threshold=args.threshold,
    )

    result = dedup.deduplicate(documents)

    # 保存结果
    output_path = os.path.join(args.input_dir, "dedup_result.json")
    with open(output_path, "w") as f:
        json.dump(result["stats"], f, indent=2)
    print(f"\n结果保存到: {output_path}")


if __name__ == "__main__":
    main()
