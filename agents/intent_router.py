"""
意图识别路由器 — 关键词快速通道 + LLM 精确判断

两级路由策略：
  1. 关键词快速匹配（0 延迟，覆盖 80% 常见场景）
  2. LLM 精确判断（with_structured_output 强制输出固定格式）

面试考点：
  Q: "怎么判断用户意图？"
  A: 两级方案。关键词快通道覆盖高频场景，0 延迟。剩下的走 LLM 精确判断，
     用 with_structured_output 约束输出，不依赖 LLM 的"自觉"。

  Q: "为什么不用纯规则或纯 LLM？"
  A: 纯规则维护成本高、覆盖不全。纯 LLM 每次调 API 有延迟和成本。
     两级结合：快的归规则，难的归模型。

Author: 程响
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import config
from app.llm_factory import get_chat_model
from app.logger import logger


# ============================================================
# 意图定义
# ============================================================

class Intent(BaseModel):
    """结构化意图输出 — with_structured_output 强制 LLM 返回此格式"""
    intent: str = Field(
        description="意图类型，只能是以下三种之一：qa（知识问答）、diagnosis（故障诊断）、report（报告生成）"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="置信度，0.0-1.0"
    )
    reason: str = Field(
        default="",
        description="判断依据，一句话说明为什么判定为此意图"
    )


# ============================================================
# 关键词快速通道
# ============================================================

# 规则格式：{意图: ([关键词列表], 置信度)}
# 顺序很重要！project_intro 必须在 diagnosis 前，防止"故障诊断系统"被误判
# 检测顺序：项目/简历 → 报告 → 诊断 → QA（QA 兜底）
FAST_ROUTES: dict[str, tuple[list[str], float]] = {
    "project_intro": (
        ["项目经历", "项目是", "项目名称", "介绍一下简历", "详细展示",
         "简历", "科研经历", "个人介绍", "我的项目", "自我介绍",
         "我叫", "我是谁", "学历", "教育背景", "实习经历",
         "专业技能", "获奖", "证书", "论文", "出版物",
         "智能运维", "故障诊断系统", "Agent Workshop", "OnCall"],
        0.95,
    ),
    "report": (
        ["生成报告", "周报", "月报", "汇总", "总结", "统计", "导出", "报表",
         "使用报告", "分析报告", "诊断报告", "情况报告", "月度", "每周"],
        0.90,
    ),
    "diagnosis": (
        ["告警", "故障", "异常", "报错", "超时", "宕机", "挂了", "不通",
         "丢包", "重启", "连接失败", "网络中断", "不可用",
         "CPU飙", "CPU 飙", "内存耗尽", "磁盘满了", "排查", "诊断",
         "flapping", "IDLE", "Idle"],  # 移除 502/503/500（HTTP 状态码属 QA），移除裸 CPU/内存/磁盘（太泛）
        0.90,
    ),
    "qa": (
        ["怎么", "什么是", "如何", "为什么", "介绍一下", "说明", "解释",
         "配置", "参数", "端口", "协议", "版本", "支持", "兼容",
         "步骤", "教程", "指南", "文档", "手册", "帮助",
         "详细介绍", "详细描述", "请介绍", "请描述", "请说明",
         "讲一下", "说说", "聊聊", "展示", "列举", "列出来",
         "写一下", "写一个", "介绍一下",
         "HTTP", "状态码", "区别", "概念", "定义", "意思", "含义",
         "怎么排查", "如何排查", "怎样排查", "如何诊断"],  # 长短语优先于 diagnosis 的"排查"
        0.85,
    ),
}


def _keyword_match(query: str) -> Optional[Intent]:
    """关键词快速匹配，收集所有命中，处理冲突后返回"""
    query_lower = query.lower()

    # 收集所有命中
    hits: list[tuple[str, str, float]] = []  # (intent, keyword, confidence)
    for intent_name, (keywords, confidence) in FAST_ROUTES.items():
        for kw in keywords:
            if kw.lower() in query_lower:
                hits.append((intent_name, kw, confidence))

    if not hits:
        return None

    # 只有一个命中 → 直接返回
    if len(hits) == 1:
        intent_name, kw, confidence = hits[0]
        logger.info(f"[IntentRouter] 关键词命中: '{kw}' → intent={intent_name}")
        return Intent(intent=intent_name, confidence=confidence, reason=f"关键词匹配: {kw}")

    # 多个命中 → 冲突解决
    intents = {h[0] for h in hits}

    # project_intro 优先级最高
    if "project_intro" in intents:
        h = next(h for h in hits if h[0] == "project_intro")
        logger.info(f"[IntentRouter] 多命中→project_intro 优先: {[h[1] for h in hits]}")
        return Intent(intent="project_intro", confidence=0.95, reason=f"多关键词命中，project_intro 优先")

    # QA vs Diagnosis 冲突 → 检查是否是知识问答句式
    if "qa" in intents and "diagnosis" in intents:
        question_patterns = ["什么是", "怎么", "如何", "区别", "为什么", "介绍一下",
                            "http", "状态码", "意思", "含义", "定义", "概念"]
        # 长 QA 短语（3字+）算 2 个信号
        qa_signals = sum(1 for p in question_patterns if p in query_lower)
        qa_long_phrases = sum(1 for h in hits if h[0] == "qa" and len(h[1]) >= 3)
        qa_signals += qa_long_phrases  # 每个长短语 +1 信号
        diag_signals = sum(1 for h in hits if h[0] == "diagnosis")

        if qa_signals >= 2:
            logger.info(f"[IntentRouter] QA/Diagnosis 冲突→QA（知识句式: {qa_signals} 个信号）")
            return Intent(intent="qa", confidence=0.80,
                         reason=f"QA/Diagnosis 冲突，知识句式信号={qa_signals}")
        if diag_signals >= 2:
            logger.info(f"[IntentRouter] QA/Diagnosis 冲突→Diagnosis（{diag_signals} 个诊断关键词）")
            return Intent(intent="diagnosis", confidence=0.80,
                         reason=f"QA/Diagnosis 冲突，诊断信号={diag_signals}")

        # 无法判断 → 降级到 L2
        logger.info(f"[IntentRouter] QA/Diagnosis 冲突无法判断，升级到 L2")
        return None

    # Report vs QA 冲突 → report 优先
    if "report" in intents:
        h = next(h for h in hits if h[0] == "report")
        return Intent(intent="report", confidence=0.90, reason="report 关键词命中")

    # 其余：取置信度最高的
    best = max(hits, key=lambda h: h[2])
    logger.info(f"[IntentRouter] 多命中→取最高置信度: '{best[1]}' → intent={best[0]}")
    return Intent(intent=best[0], confidence=best[2], reason=f"多关键词匹配: {best[1]}")


# ============================================================
# 第二层：轻量模型 + 上下文
# ============================================================

CONTEXT_AWARE_PROMPT = """你是意图识别助手。结合对话上下文判断用户意图。

上下文：{context}
用户当前消息：{query}

意图定义：
- qa：知识问答（问概念、配置、操作方法）
- diagnosis：故障诊断（描述异常、告警、故障现象）
- report：报告生成（要求生成/汇总/导出）

判断规则：
- 用户之前说"CPU告警了"，现在说"再帮我看看内存" → 仍然是 diagnosis
- 用户之前问"怎么配置"，现在说"再推荐一个方案" → 仍然是 qa
- 用户突然换了完全不同的话题 → 重新判断，降低置信度

请直接输出JSON格式的意图判断结果。"""


async def _context_aware_match(query: str, history: list[str] = None) -> Optional[Intent]:
    """第二层：轻量模型 + 对话上下文分析"""
    llm = get_chat_model(model="qwen-turbo", temperature=0.0, streaming=False)

    context = "（无历史对话）"
    if history:
        context = "\n".join(history[-4:])  # 最近4轮对话

    # 第二层：轻量模型 + 上下文
    # 注意：prompt 里的 JSON 示例用 {{ }} 转义，避免和 Python .format() 冲突
    prompt = CONTEXT_AWARE_PROMPT.format(context=context, query=query)
    chain = llm.with_structured_output(Intent)

    try:
        result = await chain.ainvoke(prompt)
        logger.info(
            f"[IntentRouter L2] 上下文分析: intent={result.intent}, "
            f"confidence={result.confidence:.2f}"
        )
        # 置信度够高才返回，否则扔给第三层
        if result.confidence >= 0.75:
            return result
        logger.info(f"[IntentRouter L2] 置信度过低({result.confidence:.2f})，升级到 L3")
        return None
    except Exception as e:
        logger.warning(f"[IntentRouter L2] 失败，升级到 L3: {e}")
        return None


# ============================================================
# 第三层：大模型精确判断
# ============================================================

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个意图识别专家，输出 JSON 格式。分析用户输入，判断其意图类型。

意图类型定义：
- qa（知识问答）：用户在询问知识、概念、操作方法、配置参数等。想获取信息而非解决问题。
  例："什么是 OSPF 协议？"、"交换机端口怎么配置？"、"这个参数什么意思？"

- diagnosis（故障诊断）：用户描述了系统异常、故障现象，需要排查问题根因。
  例："核心交换机 CPU 飙到 95%"、"服务器连不上了"、"网络延迟突然增大"

- report（报告生成）：用户要求生成报告、汇总数据、导出内容。
  例："生成本周网络运行报告"、"汇总一下最近的告警"、"给我一份月度总结"

判断规则：
1. 如果用户描述了具体异常/故障 → diagnosis
2. 如果用户要求生成/导出/汇总 → report
3. 如果用户在问知识/概念/操作/配置 → qa
4. 模糊时选最接近的，并降低置信度
"""),
    ("user", "请判断以下用户输入的意图：{query}"),
])


async def _llm_match(query: str) -> Intent:
    """LLM 精确意图判断"""
    llm = get_chat_model(temperature=0.0, streaming=False)
    chain = INTENT_PROMPT | llm.with_structured_output(Intent)

    try:
        result = await chain.ainvoke({"query": query})
        logger.info(
            f"[IntentRouter] LLM 判断: intent={result.intent}, "
            f"confidence={result.confidence:.2f}, reason={result.reason}"
        )
        return result
    except Exception as e:
        logger.warning(f"[IntentRouter] LLM 判断失败，降级为 qa: {e}")
        return Intent(
            intent="qa",
            confidence=0.5,
            reason=f"LLM 调用失败，降级为 qa: {e}",
        )


# ============================================================
# 路由器主入口
# ============================================================

@dataclass
class RouteResult:
    """路由结果"""
    intent: str           # qa / diagnosis / report
    confidence: float     # 0.0 - 1.0
    matched_by: str       # "keyword" | "llm"
    reason: str

    @property
    def target_agent(self) -> str:
        """意图 → Agent 模式映射。Auto 模式统一走 Supervisor（Boss Agent），
        Supervisor 内部会根据意图动态派发子Agent。"""
        return "supervisor"


class IntentRouter:
    """
    意图路由器

    使用：
        router = IntentRouter()
        result = await router.route("核心交换机 CPU 飙到 95%")
        # result.intent → "diagnosis"
        # result.target_agent → "plan_execute"
    """

    async def route(self, query: str, history: list[str] = None) -> RouteResult:
        """
        三层意图识别：
          L1: 关键词规则（0延迟，覆盖 ~60% 场景）
          L2: 轻量模型 + 上下文（qwen-turbo，覆盖 ~25% 场景）
          L3: 大模型精确判断（qwen-max，覆盖 ~15% 复杂场景）
        """
        if not query or not query.strip():
            return RouteResult("qa", 0.5, "default", "输入为空")

        # L1：关键词快速通道
        intent = _keyword_match(query)
        # 如果是关于简历/项目的问题，强制走 qa（防止"智能运维系统"误判为 diagnosis）
        if intent and intent.intent == "project_intro":
            return RouteResult("qa", 0.95, "keyword", "简历相关，强制 qa")
        if intent and intent.confidence >= 0.85:
            return self._to_result(intent, "keyword")

        # L2：轻量模型 + 上下文
        intent = await _context_aware_match(query, history)
        if intent:
            return self._to_result(intent, "lightweight_llm")

        # L3：大模型精确判断
        intent = await _llm_match(query)
        return self._to_result(intent, "full_llm")

    def _to_result(self, intent: Intent, matched_by: str) -> RouteResult:
        return RouteResult(
            intent=intent.intent,
            confidence=intent.confidence,
            matched_by=matched_by,
            reason=intent.reason,
        )


# 全局单例
intent_router = IntentRouter()


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        router = IntentRouter()
        queries = [
            "核心交换机 CPU 飙到 95% 怎么排查",
            "OSPF 协议是什么",
            "帮我生成这个月的网络运行报告",
            "服务器突然连不上了，报 502 错误",
            "端口 trunk 模式怎么配置",
            "最近告警太多了，汇总一下",
        ]
        for q in queries:
            result = await router.route(q)
            print(f"\n输入: {q}")
            print(f"  意图: {result.intent} | 置信度: {result.confidence:.2f}")
            print(f"  匹配方式: {result.matched_by} | 原因: {result.reason}")
            print(f"  → 目标Agent: {result.target_agent}")

    asyncio.run(test())
