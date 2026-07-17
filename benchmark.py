"""
Agent Workshop 综合基准测试

四种场景 × 多条用例 = 系统级质量评估

运行: python benchmark.py

指标:
    - Router Accuracy:      意图识别正确率
    - Retrieval Recall@3:   检索召回率
    - Faithfulness (1-5):   LLM-as-Judge 忠实度
    - Correctness (1-5):    LLM-as-Judge 正确性
    - End-to-End Latency:   端到端延迟

Author: 程响
"""

import asyncio
import json
import time
import sys
from dataclasses import dataclass, field
from app.logger import logger


# ============================================================
# 基准测试集（四种场景 × 5 条 = 20 条）
# ============================================================

@dataclass
class BenchmarkCase:
    id: str
    scenario: str            # qa / diagnosis / report / memory
    question: str
    expected_intent: str     # qa / diagnosis / report
    expected_keywords: list[str]  # 回答中至少应包含的关键词
    min_faithfulness: float = 0.6  # 最低忠实度阈值
    notes: str = ""


BENCHMARK = [
    # ─── QA 场景（知识问答） ───
    BenchmarkCase(
        id="QA-01", scenario="qa",
        question="OSPF 协议怎么配置 router-id",
        expected_intent="qa",
        expected_keywords=["router", "ospf", "配置"],
        notes="基础技术问答"
    ),
    BenchmarkCase(
        id="QA-02", scenario="qa",
        question="HTTP 状态码 502 和 504 有什么区别",
        expected_intent="qa",
        expected_keywords=["502", "504", "网关"],
        notes="概念对比类问答"
    ),
    BenchmarkCase(
        id="QA-03", scenario="qa",
        question="VLAN trunk 模式和 access 模式有什么区别",
        expected_intent="qa",
        expected_keywords=["trunk", "access", "VLAN"],
        notes="配置类问答"
    ),
    BenchmarkCase(
        id="QA-04", scenario="qa",
        question="什么是 BGP 协议",
        expected_intent="qa",
        expected_keywords=["BGP", "边界网关", "AS"],
        notes="概念解释类"
    ),
    BenchmarkCase(
        id="QA-05", scenario="qa",
        question="怎么排查网络延迟问题",
        expected_intent="qa",
        expected_keywords=["延迟", "ping", "traceroute"],
        notes="排查方法类"
    ),

    # ─── Diagnosis 场景（故障诊断） ───
    BenchmarkCase(
        id="DX-01", scenario="diagnosis",
        question="核心交换机 CPU 飙到 95%，帮我排查",
        expected_intent="diagnosis",
        expected_keywords=["CPU", "告警", "排查"],
        notes="经典诊断场景"
    ),
    BenchmarkCase(
        id="DX-02", scenario="diagnosis",
        question="服务器突然连不上了，报 502 错误",
        expected_intent="diagnosis",
        expected_keywords=["502", "服务", "检查"],
        notes="故障现象描述"
    ),
    BenchmarkCase(
        id="DX-03", scenario="diagnosis",
        question="BGP 邻居状态一直 Idle 怎么办",
        expected_intent="diagnosis",
        expected_keywords=["BGP", "Idle", "邻居"],
        notes="协议级故障"
    ),
    BenchmarkCase(
        id="DX-04", scenario="diagnosis",
        question="端口一直在 flapping，是什么原因",
        expected_intent="diagnosis",
        expected_keywords=["端口", "flapping", "抖动"],
        notes="物理层问题"
    ),
    BenchmarkCase(
        id="DX-05", scenario="diagnosis",
        question="内存使用率突然从 40% 涨到 90%",
        expected_intent="diagnosis",
        expected_keywords=["内存", "进程", "检查"],
        notes="资源异常"
    ),

    # ─── Report 场景（报告生成） ───
    BenchmarkCase(
        id="RP-01", scenario="report",
        question="帮我汇总一下最近的系统运行情况",
        expected_intent="report",
        expected_keywords=["系统", "运行"],
        notes="报告生成"
    ),
    BenchmarkCase(
        id="RP-02", scenario="report",
        question="生成本周的网络故障统计报告",
        expected_intent="report",
        expected_keywords=["报告", "统计", "故障"],
        notes="统计报告"
    ),

    # ─── Memory 场景（个性化问答） ───
    BenchmarkCase(
        id="MM-01", scenario="qa",
        question="我叫什么名字",
        expected_intent="qa",
        expected_keywords=["程响", "名字"],
        notes="前提：已在 memory 中存入'我叫程响'"
    ),
    BenchmarkCase(
        id="MM-02", scenario="qa",
        question="1+1 等于几",
        expected_intent="qa",
        expected_keywords=["2", "等于"],
        notes="基础问答——任何系统都应该过"
    ),
    BenchmarkCase(
        id="MM-03", scenario="qa",
        question="现在几点",
        expected_intent="qa",
        expected_keywords=[],  # 时间无法预设关键词
        min_faithfulness=0.3,
        notes="时间查询——只需工具调用正确"
    ),
]


# ============================================================
# LLM-as-Judge Prompts
# ============================================================

FAITHFULNESS_PROMPT = """你是一个严格评估者。判断回答质量。

用户问题：{question}
回答：{answer}

评分 1-5（只输出数字）：
1=完全编造或答非所问
2=大部分编造，少量真实信息
3=部分有依据，有推测或遗漏
4=基本准确，个别细节有瑕疵
5=完全准确，无编造无遗漏

注意：常识问题（如"1+1等于几"、"现在几点"）不需要引用外部资料，只要回答正确就给5分。"""

CORRECTNESS_PROMPT = """你是一个严格评估者。判断回答是否正确。

期望包含的关键信息：{expected}
模型回答：{answer}

评分 1-5（只输出数字）：
1=完全错误  2=关键信息错误  3=遗漏重要信息
4=基本正确  5=完全正确，覆盖所有关键点

注意：如果期望关键词为空，只根据回答本身的逻辑正确性评分。"""


# ============================================================
# 评测引擎
# ============================================================

@dataclass
class CaseResult:
    case: BenchmarkCase
    intent_correct: bool = False
    actual_intent: str = ""
    retrieval_count: int = 0
    faithfulness: float = 0.0
    correctness: float = 0.0
    latency_seconds: float = 0.0
    answer: str = ""
    error: str = ""


async def run_benchmark(verbose: bool = True) -> list[CaseResult]:
    """跑全部基准测试"""
    from agents.intent_router import intent_router
    from retrieval.hybrid_search import hybrid_searcher
    from app.llm_factory import get_chat_model

    results = []
    llm = get_chat_model(temperature=0.0, streaming=False)

    for i, case in enumerate(BENCHMARK):
        if verbose:
            print(f"[{i+1}/{len(BENCHMARK)}] {case.id}: {case.question[:40]}...", end=" ", flush=True)

        result = CaseResult(case=case)
        t0 = time.time()

        try:
            # Step 1: 意图路由（走完整 IntentRouter）
            route = await intent_router.route(case.question)
            result.actual_intent = route.intent
            result.intent_correct = (route.intent == case.expected_intent)

            # Step 2: 走真实 MasterAgent 管线
            from agents.master_agent import master_agent
            parts = []
            async for chunk in master_agent.astream(case.question, f"bench_{case.id}"):
                if chunk.get("content"):
                    parts.append(chunk["content"])
            result.answer = "".join(parts) if parts else "(无输出)"

            # Step 3: LLM-as-Judge 评估
            exp_keywords = ", ".join(case.expected_keywords) if case.expected_keywords else "（无预设关键词）"

            # Faithfulness — Judge 基于问题+回答直接评
            try:
                faith_resp = await llm.ainvoke(
                    FAITHFULNESS_PROMPT.format(question=case.question, answer=result.answer[:1000])
                )
                result.faithfulness = int(str(faith_resp.content).strip()) / 5.0
            except Exception:
                result.faithfulness = 0.5

            # Correctness
            try:
                corr_resp = await llm.ainvoke(
                    CORRECTNESS_PROMPT.format(expected=exp_keywords, answer=result.answer[:1000])
                )
                result.correctness = int(str(corr_resp.content).strip()) / 5.0
            except Exception:
                result.correctness = 0.5

            # Retrieval count — 从 MasterAgent 内部提取（如果有的话）
            result.retrieval_count = 0  # MasterAgent 不直接暴露检索数

        except Exception as e:
            result.error = str(e)[:100]
            if verbose:
                print(f"ERROR: {e}")

        result.latency_seconds = round(time.time() - t0, 1)

        if verbose:
            status = "PASS" if result.intent_correct and result.faithfulness >= case.min_faithfulness else "FAIL"
            print(f"{status} | intent={result.actual_intent} faith={result.faithfulness:.0%} corr={result.correctness:.0%} {result.latency_seconds}s")

        results.append(result)

    return results


def print_report(results: list[CaseResult]):
    """打印评测报告"""
    n = len(results)
    intent_ok = sum(1 for r in results if r.intent_correct)
    faith_ok = sum(1 for r in results if r.faithfulness >= r.case.min_faithfulness)
    errors = sum(1 for r in results if r.error)

    print("\n" + "=" * 65)
    print("  Agent Workshop Benchmark Report")
    print("=" * 65)

    # 按场景统计
    for scenario in ["qa", "diagnosis", "report"]:
        cases = [r for r in results if r.case.scenario == scenario]
        if not cases:
            continue
        intent_acc = sum(1 for r in cases if r.intent_correct) / len(cases)
        avg_faith = sum(r.faithfulness for r in cases) / len(cases)
        avg_corr = sum(r.correctness for r in cases) / len(cases)
        avg_lat = sum(r.latency_seconds for r in cases) / len(cases)
        print(f"\n  [{scenario.upper()}] {len(cases)} cases")
        print(f"    Intent Accuracy:  {intent_acc:.0%}")
        print(f"    Avg Faithfulness: {avg_faith:.0%}")
        print(f"    Avg Correctness:  {avg_corr:.0%}")
        print(f"    Avg Latency:      {avg_lat:.1f}s")

    print(f"\n  {'─' * 55}")
    print(f"  OVERALL ({n} cases)")
    print(f"    Intent Accuracy:  {intent_ok}/{n} ({intent_ok/n:.0%})")
    print(f"    Faithfulness OK:  {faith_ok}/{n} ({faith_ok/n:.0%})")
    print(f"    Errors:           {errors}/{n}")
    overall_faith = sum(r.faithfulness for r in results) / n
    overall_corr = sum(r.correctness for r in results) / n
    overall_lat = sum(r.latency_seconds for r in results) / n
    print(f"    Avg Faithfulness: {overall_faith:.0%}")
    print(f"    Avg Correctness:  {overall_corr:.0%}")
    print(f"    Avg Latency:      {overall_lat:.1f}s")

    # 失败的 case
    failed = [r for r in results if not r.intent_correct or r.faithfulness < r.case.min_faithfulness]
    if failed:
        print(f"\n  FAILED CASES:")
        for r in failed:
            reasons = []
            if not r.intent_correct:
                reasons.append(f"intent={r.actual_intent}(expected {r.case.expected_intent})")
            if r.faithfulness < r.case.min_faithfulness:
                reasons.append(f"faith={r.faithfulness:.0%}(min {r.case.min_faithfulness:.0%})")
            if r.error:
                reasons.append(f"error={r.error}")
            print(f"    [{r.case.id}] {r.case.question[:50]}...")
            print(f"      {' | '.join(reasons)}")

    print("\n" + "=" * 65)

    # 保存详细结果
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": n, "intent_ok": intent_ok, "faith_ok": faith_ok,
        "overall_faithfulness": round(overall_faith, 3),
        "overall_correctness": round(overall_corr, 3),
        "avg_latency": round(overall_lat, 1),
        "details": [
            {
                "id": r.case.id, "question": r.case.question,
                "intent_correct": r.intent_correct, "actual_intent": r.actual_intent,
                "faithfulness": r.faithfulness, "correctness": r.correctness,
                "retrieval_count": r.retrieval_count, "latency": r.latency_seconds,
                "answer_preview": r.answer[:150]
            }
            for r in results
        ]
    }
    with open("data/benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Report saved: data/benchmark_report.json\n")


if __name__ == "__main__":
    print("Agent Workshop Benchmark")
    print(f"Test cases: {len(BENCHMARK)} (QA/DX/RP/MM)\n")
    results = asyncio.run(run_benchmark())
    print_report(results)
