"""
Radix Attention 原理演示

SGLang 的核心创新: 用 Radix Tree 管理 KV Cache 前缀。
自动发现和共享公共前缀，无需用户手动配置。
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class RadixNode:
    """Radix Tree 的节点"""
    # 这个节点存储的 token 序列片段
    tokens: List[int] = field(default_factory=list)
    # 子节点: first_token -> RadixNode
    children: Dict[int, 'RadixNode'] = field(default_factory=dict)
    # 对应的 KV Cache Block IDs (如果已缓存)
    kv_block_ids: List[int] = field(default_factory=list)
    # 引用计数
    ref_count: int = 0
    # 最后访问时间 (用于 LRU 驱逐)
    last_access_time: int = 0


class RadixTree:
    """
    Radix Tree — SGLang 用于管理 KV Cache 前缀的数据结构

    思想: 用 Radix Tree 存储所有活跃序列的 token 前缀。
    - 公共前缀自动合并 → 共享 KV Cache
    - 查询新序列时，沿树查找最长匹配前缀
    - 匹配的部分: 复用 KV Cache (跳过 Prefill!)
    - 不匹配的部分: 只 Prefill 剩余 token
    """

    def __init__(self):
        self.root = RadixNode()
        self.time_counter = 0

    def insert(self, tokens: List[int], kv_block_ids: List[int]):
        """
        插入一个 token 序列和对应的 KV Cache Block IDs

        如果前缀已存在，只需要存储新增部分。
        """
        self.time_counter += 1
        node = self.root

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token in node.children:
                child = node.children[token]
                # 匹配这个子节点的 tokens
                match_len = 0
                while (match_len < len(child.tokens) and
                       i + match_len < len(tokens) and
                       child.tokens[match_len] == tokens[i + match_len]):
                    match_len += 1

                if match_len == len(child.tokens):
                    # 完全匹配，继续往下
                    child.ref_count += 1
                    child.last_access_time = self.time_counter
                    i += match_len
                    node = child
                else:
                    # 部分匹配，需要分裂节点
                    self._split_node(child, match_len)
                    child.ref_count += 1
                    child.last_access_time = self.time_counter
                    i += match_len
                    node = child
            else:
                # 无匹配，创建新节点
                remaining = tokens[i:]
                remaining_kvs = kv_block_ids[i:] if i < len(kv_block_ids) else []
                new_node = RadixNode(
                    tokens=remaining,
                    kv_block_ids=remaining_kvs,
                    ref_count=1,
                    last_access_time=self.time_counter,
                )
                node.children[token] = new_node
                return

    def _split_node(self, node: RadixNode, split_pos: int):
        """分裂节点: 在 split_pos 处将节点一分为二"""
        # 创建下半部分
        suffix_tokens = node.tokens[split_pos:]
        suffix_node = RadixNode(
            tokens=suffix_tokens,
            children=node.children,
            kv_block_ids=node.kv_block_ids[split_pos:] if split_pos < len(node.kv_block_ids) else [],
            ref_count=node.ref_count,
            last_access_time=node.last_access_time,
        )

        # 更新当前节点为上半部分
        node.tokens = node.tokens[:split_pos]
        node.kv_block_ids = node.kv_block_ids[:split_pos]
        node.children = {suffix_tokens[0]: suffix_node} if suffix_tokens else {}

    def match_prefix(self, tokens: List[int]) -> Tuple[int, List[int]]:
        """
        查找最长匹配前缀

        Returns:
            (matched_length, matched_kv_block_ids)
        """
        self.time_counter += 1
        node = self.root
        matched_len = 0
        matched_kvs = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token not in node.children:
                break

            child = node.children[token]
            # 匹配子节点的 tokens
            match_len = 0
            while (match_len < len(child.tokens) and
                   i + match_len < len(tokens) and
                   child.tokens[match_len] == tokens[i + match_len]):
                match_len += 1

            matched_len += match_len
            matched_kvs.extend(child.kv_block_ids[:match_len])
            child.last_access_time = self.time_counter

            if match_len < len(child.tokens):
                break  # 部分匹配

            i += match_len
            node = child

        return matched_len, matched_kvs

    def print_tree(self, node: RadixNode = None, prefix: str = "", depth: int = 0):
        """可视化打印 Radix Tree"""
        if node is None:
            node = self.root
            print("  Radix Tree:")

        for token, child in node.children.items():
            tokens_str = str(child.tokens[:10])
            if len(child.tokens) > 10:
                tokens_str += "..."
            indent = "  " + "│ " * depth
            print(f"{indent}├─ {tokens_str} (refs={child.ref_count}, kvs={len(child.kv_block_ids)})")
            self.print_tree(child, prefix, depth + 1)


def demo_radix_attention():
    """演示 Radix Attention 的工作原理"""
    print("\n" + "=" * 70)
    print("  Radix Attention Demo")
    print("=" * 70)

    tree = RadixTree()

    # 模拟 token IDs
    system_prompt = [100, 101, 102, 103, 104, 105, 106, 107]  # "You are helpful..."
    user_a = [200, 201, 202]       # "What is AI?"
    user_b = [300, 301, 302, 303]  # "Write code for me"
    user_c = [200, 201, 400]       # "What is ML?" (共享 "What is" 前缀)

    # 请求 1: System Prompt + User A
    tokens_1 = system_prompt + user_a
    kvs_1 = list(range(len(tokens_1)))  # 模拟 KV Block IDs

    print(f"\n  [Request 1] System + 'What is AI?' ({len(tokens_1)} tokens)")
    matched, matched_kvs = tree.match_prefix(tokens_1)
    print(f"  Prefix match: {matched} tokens (cache {'HIT' if matched > 0 else 'MISS'})")
    print(f"  Need to Prefill: {len(tokens_1) - matched} tokens")
    tree.insert(tokens_1, kvs_1)

    # 请求 2: System Prompt + User B (共享 System Prompt!)
    tokens_2 = system_prompt + user_b
    kvs_2 = list(range(len(tokens_2)))

    print(f"\n  [Request 2] System + 'Write code' ({len(tokens_2)} tokens)")
    matched, matched_kvs = tree.match_prefix(tokens_2)
    print(f"  Prefix match: {matched} tokens (cache HIT for system prompt!)")
    print(f"  Need to Prefill: {len(tokens_2) - matched} tokens only")
    print(f"  Saved: {matched} tokens of Prefill computation!")
    tree.insert(tokens_2, kvs_2)

    # 请求 3: System Prompt + User C (共享 System + "What is")
    tokens_3 = system_prompt + user_c
    kvs_3 = list(range(len(tokens_3)))

    print(f"\n  [Request 3] System + 'What is ML?' ({len(tokens_3)} tokens)")
    matched, matched_kvs = tree.match_prefix(tokens_3)
    print(f"  Prefix match: {matched} tokens")
    print(f"  (System prompt + 'What is' prefix cached!)")
    print(f"  Need to Prefill: {len(tokens_3) - matched} tokens only")
    tree.insert(tokens_3, kvs_3)

    # 打印 Radix Tree
    print(f"\n  Current Radix Tree state:")
    tree.print_tree()

    # 对比分析
    print(f"\n  {'='*60}")
    print(f"  Savings Analysis:")
    print(f"  {'='*60}")
    total_tokens = len(tokens_1) + len(tokens_2) + len(tokens_3)
    without_cache = total_tokens  # 朴素方案: 每个请求全部 Prefill
    # 请求 1: 全部 Prefill, 请求 2: 跳过 system, 请求 3: 跳过 system+"What is"
    with_cache = len(tokens_1) + len(user_b) + (len(user_c) - 2)  # 近似
    print(f"  Without Radix Attention: {without_cache} tokens to Prefill")
    print(f"  With Radix Attention:    ~{with_cache} tokens to Prefill")
    print(f"  Savings: {(1 - with_cache/without_cache)*100:.0f}% less Prefill compute!")


if __name__ == "__main__":
    demo_radix_attention()

    print("\n" + "=" * 70)
    print("  Key Takeaways:")
    print("  1. Radix Tree automatically discovers shared prefixes")
    print("  2. No manual configuration needed (unlike vLLM prefix caching)")
    print("  3. Works across requests, multi-turn conversations")
    print("  4. Saves both compute (skip Prefill) and memory (shared KV)")
    print("=" * 70)
