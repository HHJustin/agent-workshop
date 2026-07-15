"""
Memory Benchmark — 评估长期记忆检索质量

指标：召回准确率 (Recall) / 记忆中利用率 (Memory Utilization)

使用：
    python -m memory.benchmark

Author: 程响
"""

import asyncio
import json
import time
from dataclasses import dataclass, field

TEST_QUERIES = [
    # (用户问题, 期望召回的记忆关键词)
    ("我叫什么名字", ["程响", "名字"]),
    ("我之前提到的项目叫什么", ["Agent Workshop", "智能运维"]),
    ("我最常用的编程语言是什么", ["Python"]),
    ("我的邮箱地址是什么", ["chengx0409", "163.com"]),
    ("我做过的项目涉及什么技术栈", ["LangChain", "FastAPI", "Milvus"]),
]


@dataclass
class BenchmarkResult:
    recall: float = 0.0          # 召回率（期望关键词命中的比例）
    memory_used: int = 0         # 检索到的记忆数
    total_memories: int = 0      # 用户总记忆数
    latency_ms: float = 0.0      # 平均检索延迟
    per_query: list[dict] = field(default_factory=list)

    @property
    def utilization(self) -> float:
        """记忆中利用率 = 检索到的记忆 / 总记忆"""
        return self.memory_used / max(self.total_memories, 1)


async def run_benchmark(user_id: str = "benchmark_user") -> BenchmarkResult:
    """运行记忆检索 Benchmark"""
    from memory.store import MemoryStore

    store = MemoryStore()
    result = BenchmarkResult()
    result.total_memories = store.stats(user_id)["total"]

    for query, expected_keywords in TEST_QUERIES:
        t0 = time.time()
        try:
            memories = store.search(user_id, query, limit=5)
        except Exception:
            memories = store.search_fallback(user_id, query, limit=5)
        elapsed = (time.time() - t0) * 1000

        # 计算召回率
        recalled_text = " ".join(
            m.summary + " " + m.keywords + " " + m.content[:100]
            for m in memories
        ).lower()

        matched = sum(1 for kw in expected_keywords if kw.lower() in recalled_text)
        recall = matched / len(expected_keywords) if expected_keywords else 0

        result.per_query.append({
            "query": query,
            "expected_kw": expected_keywords,
            "recalled": len(memories),
            "matched_kw": matched,
            "recall": recall,
            "latency_ms": round(elapsed, 1),
        })
        result.memory_used += len(memories)
        result.latency_ms += elapsed

    if result.per_query:
        result.recall = sum(q["recall"] for q in result.per_query) / len(result.per_query)
    result.latency_ms = result.latency_ms / max(len(result.per_query), 1)

    return result


def print_report(result: BenchmarkResult):
    """打印 Benchmark 报告"""
    print("\n" + "=" * 60)
    print("Memory Benchmark Report")
    print("=" * 60)
    print(f"  Total memories:     {result.total_memories}")
    print(f"  Avg recall:         {result.recall:.1%}")
    print(f"  Avg utilization:    {result.utilization:.1%}")
    print(f"  Avg latency:        {result.latency_ms:.1f} ms")
    print()
    for q in result.per_query:
        status = "PASS" if q["recall"] >= 0.5 else "FAIL"
        print(f"  [{status}] {q['query']}")
        print(f"    recalled={q['recalled']}, matched={q['matched_kw']}/{len(q['expected_kw'])}, "
              f"recall={q['recall']:.0%}, {q['latency_ms']:.0f}ms")


if __name__ == "__main__":
    print("Memory Benchmark")
    print("(先运行 app 产生一些记忆数据，再跑 benchmark)")
    result = asyncio.run(run_benchmark())
    print_report(result)
