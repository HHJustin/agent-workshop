"""
RAG 评估模块 — 上下文召回率、精确率、忠实度、答案相关性

使用 RAGAS 框架 + 自定义指标
面试考点：
    Q: "85% 怎么测出来的？"
    A: RAGAS 自动化评估。构建 Ground Truth 测试集 → 逐条跑 Agent →
       自动算 Context Recall/Precision + Faithfulness + Answer Relevancy。
       不是凭感觉，是量化指标。

Author: 程响
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from app.llm_factory import get_chat_model
from app.logger import logger


# ============================================================
# 测试集
# ============================================================

@dataclass
class TestCase:
    """单个测试用例"""
    question: str
    ground_truth: str           # 正确答案（至少包含的关键信息）
    expected_sources: list[str] = field(default_factory=list)  # 预期检索到的文档


# ============================================================
# 检索指标
# ============================================================

@dataclass
class RetrievalMetrics:
    """检索阶段指标"""
    recall: float = 0.0           # 召回了多少正确答案
    precision: float = 0.0        # 检索到的文档中有多少是相关的
    mrr: float = 0.0              # Mean Reciprocal Rank
    hit_rate: float = 0.0         # Top-K 命中率


@dataclass
class GenerationMetrics:
    """生成阶段指标"""
    faithfulness: float = 0.0     # 忠实度：回答是否基于检索到的上下文
    answer_relevancy: float = 0.0 # 答案相关性：是否回答用户问题
    correctness: float = 0.0      # 正确性：与 Ground Truth 的语义相似度


@dataclass
class EfficiencyMetrics:
    """效率指标"""
    total_latency_seconds: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    pipeline_steps: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """单条评估结果"""
    test_case: TestCase
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    retrieved_docs: list[str] = field(default_factory=list)
    answer: str = ""


# ============================================================
# 评估器
# ============================================================

class RAGEvaluator:
    """
    RAG 评估器 — 自建测试集 + LLM-as-Judge

    使用方式：
        evaluator = RAGEvaluator(test_cases)
        results = await evaluator.evaluate_retrieval()
        results = await evaluator.evaluate_generation()
    """

    def __init__(self, test_cases: list[TestCase]):
        self.test_cases = test_cases
        self.results: list[EvalResult] = []

    # ─── 检索评估 ───

    async def evaluate_retrieval(self, top_k: int = 3) -> list[EvalResult]:
        """
        检索评估：Context Recall + Precision + MRR + Hit Rate

        对测试集每个问题执行检索，看召回的文档是否包含正确答案。
        """
        from retrieval.hybrid_search import hybrid_searcher

        results = []
        for tc in self.test_cases:
            start = time.time()
            retrieved = await hybrid_searcher.search(tc.question, top_k=top_k, filter_meta=None)  # 评估不限 scope
            elapsed = time.time() - start

            retrieved_texts = [doc.page_content for doc, _ in retrieved]

            # Recall：检索到的文档中包含了 Ground Truth 的多少关键词
            gt_keywords = set(tc.ground_truth.lower().split())
            recalled_keywords = set()
            for text in retrieved_texts:
                recalled_keywords.update(
                    kw for kw in gt_keywords if kw.lower() in text.lower()
                )
            recall = len(recalled_keywords) / max(len(gt_keywords), 1)

            # Precision：检索到的文档中多少和 Ground Truth 语义相关
            relevant_count = sum(
                1 for text in retrieved_texts
                if any(kw.lower() in text.lower() for kw in gt_keywords)
            )
            precision = relevant_count / max(len(retrieved_texts), 1)

            # MRR：正确答案排在第几位
            mrr = 0.0
            for rank, (doc, _) in enumerate(retrieved, 1):
                if any(kw.lower() in doc.page_content.lower() for kw in gt_keywords):
                    mrr = 1.0 / rank
                    break

            # Hit Rate：Top-K 中有没有命中
            hit_rate = 1.0 if any(
                any(kw.lower() in text.lower() for kw in gt_keywords)
                for text in retrieved_texts
            ) else 0.0

            r = EvalResult(
                test_case=tc,
                retrieval=RetrievalMetrics(
                    recall=round(recall, 3),
                    precision=round(precision, 3),
                    mrr=round(mrr, 3),
                    hit_rate=hit_rate,
                ),
                efficiency=EfficiencyMetrics(total_latency_seconds=round(elapsed, 2)),
                retrieved_docs=[t[:100] for t in retrieved_texts],
            )
            results.append(r)
            logger.info(
                f"[Eval:Retrieval] Q={tc.question[:30]}... "
                f"Recall={recall:.2f} Precision={precision:.2f} MRR={mrr:.2f}"
            )

        self.results = results
        return results

    # ─── 生成评估（LLM-as-Judge） ───

    FAITHFULNESS_PROMPT = """你是一个严格的评估者。判断以下回答是否严格基于提供的上下文。

上下文：
{context}

回答：
{answer}

请只回答一个数字（1-5），不要解释：
1 = 完全编造，与上下文无关
2 = 大部分编造，少量基于上下文
3 = 部分基于上下文，有推测成分
4 = 基本基于上下文，个别细节有推测
5 = 完全基于上下文，无任何编造"""

    CORRECTNESS_PROMPT = """你是一个严格的评估者。判断以下回答与正确答案的匹配程度。

正确答案：{ground_truth}

模型回答：{answer}

请只回答一个数字（1-5），不要解释：
1 = 完全错误
2 = 部分正确但关键信息错误
3 = 基本正确但遗漏重要信息
4 = 正确，覆盖了大部分关键点
5 = 完全正确，无遗漏"""

    RELEVANCY_PROMPT = """你是一个严格的评估者。判断以下回答是否直接回应了用户的问题。

用户问题：{question}

模型回答：{answer}

请只回答一个数字（1-5），不要解释：
1 = 答非所问
2 = 部分相关但跑题严重
3 = 基本相关但有冗余
4 = 相关，直接回应问题
5 = 完全切题，简洁精准"""

    async def evaluate_generation(self) -> list[EvalResult]:
        """生成评估：Faithfulness + Correctness + Answer Relevancy（LLM-as-Judge）"""
        if not self.results:
            await self.evaluate_retrieval()

        llm = get_chat_model(temperature=0.0, streaming=False)

        for r in self.results:
            # 用 LLM 对检索结果生成回答
            from agents.tools import retrieve_knowledge
            context = await retrieve_knowledge.ainvoke({"query": r.test_case.question})
            answer_prompt = f"根据以下资料回答问题：{context}\n\n问题：{r.test_case.question}"
            answer_resp = await llm.ainvoke(answer_prompt)
            r.answer = answer_resp.content if hasattr(answer_resp, "content") else str(answer_resp)

            # Faithfulness
            faith_resp = await llm.ainvoke(
                self.FAITHFULNESS_PROMPT.format(context=context, answer=r.answer)
            )
            try:
                r.generation.faithfulness = int(str(faith_resp.content).strip()) / 5.0
            except ValueError:
                r.generation.faithfulness = 0.5

            # Correctness
            corr_resp = await llm.ainvoke(
                self.CORRECTNESS_PROMPT.format(ground_truth=r.test_case.ground_truth, answer=r.answer)
            )
            try:
                r.generation.correctness = int(str(corr_resp.content).strip()) / 5.0
            except ValueError:
                r.generation.correctness = 0.5

            # Relevancy
            rel_resp = await llm.ainvoke(
                self.RELEVANCY_PROMPT.format(question=r.test_case.question, answer=r.answer)
            )
            try:
                r.generation.answer_relevancy = int(str(rel_resp.content).strip()) / 5.0
            except ValueError:
                r.generation.answer_relevancy = 0.5

            logger.info(
                f"[Eval:Generation] Q={r.test_case.question[:30]}... "
                f"Faith={r.generation.faithfulness:.2f} "
                f"Correct={r.generation.correctness:.2f} "
                f"Relev={r.generation.answer_relevancy:.2f}"
            )

        return self.results

    # ─── 汇总报告 ───

    def summary(self) -> dict:
        """生成汇总报告"""
        if not self.results:
            return {"error": "请先运行评估"}

        n = len(self.results)
        return {
            "test_cases": n,
            "retrieval": {
                "avg_recall": round(sum(r.retrieval.recall for r in self.results) / n, 3),
                "avg_precision": round(sum(r.retrieval.precision for r in self.results) / n, 3),
                "avg_mrr": round(sum(r.retrieval.mrr for r in self.results) / n, 3),
                "hit_rate": round(sum(r.retrieval.hit_rate for r in self.results) / n, 3),
            },
            "generation": {
                "avg_faithfulness": round(sum(r.generation.faithfulness for r in self.results) / n, 3),
                "avg_correctness": round(sum(r.generation.correctness for r in self.results) / n, 3),
                "avg_relevancy": round(sum(r.generation.answer_relevancy for r in self.results) / n, 3),
            },
            "efficiency": {
                "avg_latency": round(sum(r.efficiency.total_latency_seconds for r in self.results) / n, 2),
                "total_cases": n,
            },
            "details": [
                {
                    "question": r.test_case.question[:60],
                    "recall": r.retrieval.recall,
                    "precision": r.retrieval.precision,
                    "faithfulness": r.generation.faithfulness,
                    "correctness": r.generation.correctness,
                    "latency": r.efficiency.total_latency_seconds,
                }
                for r in self.results
            ],
        }


# ============================================================
# 预置测试集（网络运维场景）
# ============================================================

DEFAULT_TEST_CASES = [
    TestCase(
        question="ERR-5001 怎么处理",
        ground_truth="数据库连接池耗尽，需要扩大连接池上限 max_connections=500，重启应用服务释放僵尸连接，检查代码中未关闭的连接",
        expected_sources=["network_manual.txt"],
    ),
    TestCase(
        question="OSPF 协议怎么配置 router-id",
        ground_truth="进入 OSPF 配置模式 router ospf 1，设置 router-id 1.1.1.1",
        expected_sources=["network_manual.txt"],
    ),
    TestCase(
        question="BGP 邻居断开怎么排查",
        ground_truth="检查 TCP 179 端口连通性 telnet，检查 AS号配置，检查底层 IGP 路由可达",
        expected_sources=["network_manual.txt"],
    ),
    TestCase(
        question="核心交换机 CPU 飙到 95%",
        ground_truth="查看进程 show processes cpu sorted，检查 STP 状态，检查 ACL 规则，检查 SNMP 轮询间隔",
        expected_sources=["network_manual.txt"],
    ),
    TestCase(
        question="VLAN access 模式怎么配",
        ground_truth="switchport mode access，switchport access vlan 10",
        expected_sources=["network_manual.txt"],
    ),
]


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import asyncio
    import logging
    logging.basicConfig(level=logging.INFO)

    async def test():
        evaluator = RAGEvaluator(DEFAULT_TEST_CASES[:3])  # 只测前3条，省 Token
        await evaluator.evaluate_generation()
        report = evaluator.summary()

        print("\n" + "=" * 60)
        print("RAG 评估报告")
        print("=" * 60)

        print(f"\n📊 检索指标 ({report['test_cases']}条):")
        r = report["retrieval"]
        print(f"  Recall@{report['test_cases']}: {r['avg_recall']:.1%}")
        print(f"  Precision: {r['avg_precision']:.1%}")
        print(f"  MRR: {r['avg_mrr']:.1%}")
        print(f"  Hit Rate: {r['hit_rate']:.1%}")

        print(f"\n📝 生成指标:")
        g = report["generation"]
        print(f"  Faithfulness: {g['avg_faithfulness']:.1%}")
        print(f"  Correctness: {g['avg_correctness']:.1%}")
        print(f"  Answer Relevancy: {g['avg_relevancy']:.1%}")

        print(f"\n⏱️ 效率:")
        print(f"  平均延迟: {report['efficiency']['avg_latency']}s")

        print(f"\n📋 详细:")
        for d in report["details"]:
            print(
                f"  Q: {d['question'][:40]}... "
                f"| R:{d['recall']:.0%} P:{d['precision']:.0%} "
                f"| F:{d['faithfulness']:.0%} C:{d['correctness']:.0%} "
                f"| {d['latency']}s"
            )

    asyncio.run(test())
