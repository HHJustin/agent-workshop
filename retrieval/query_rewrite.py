"""
Query Rewrite 查询改写 — 提升检索召回率

三种策略：
  1. LLM 改写：口语→专业术语（"CPU飙了"→"CPU使用率过高"）
  2. 多路召回：生成多个变体同时检索
  3. HyDE：先让 LLM 生成假设答案，用答案检索（文档风格更接近）

面试考点：
    Q: "为什么需要 Query Rewrite？"
    A: 用户的自然语言和文档的专业表述有 gap。比如用户说"网断了"，
       文档里写的是"网络中断"、"链路故障"、"端口 down"。Query Rewrite
       把口语化 query 转成专业术语，消弭用户语言和文档语言的鸿沟。

Author: 程响
"""

from __future__ import annotations

from app.llm_factory import get_chat_model
from app.logger import logger


# ============================================================
# 策略1：LLM 改写
# ============================================================

REWRITE_PROMPT = """你是查询优化专家。将用户的口语化问题改写为更适合检索的专业查询。

核心原则：提取问题中的实体、概念、关键词，生成密集且精确的检索词。

规则：
1. 识别问题类型，选择对应策略：
   - 简历/文档类 → 提取：人名、项目名、技术栈、公司名、职位、专业术语
   - 故障/运维类 → 口语转术语："挂了"→"故障/宕机"，"慢"→"延迟/性能下降"
   - 概念/知识类 → 提取核心概念，补充同义词
2. 保留所有专有名词（项目名、产品名、协议名）不做改动
3. 输出纯关键词和短语（空格分隔），不超过 60 个字符
4. 不要输出完整句子，不要加解释

用户问题：{query}
改写结果："""


async def llm_rewrite(query: str) -> str:
    """
    用 LLM 改写用户查询

    "网断了" → "网络中断 链路故障 端口down 排查方法"
    "CPU飙了" → "CPU使用率过高 核心交换机 根因分析 解决方法"
    """
    try:
        llm = get_chat_model(model="qwen-turbo", temperature=0.0, streaming=False)
        response = await llm.ainvoke(REWRITE_PROMPT.format(query=query))
        rewritten = response.content.strip() if hasattr(response, "content") else str(response)

        if rewritten and len(rewritten) < 200:
            logger.info(f"[QueryRewrite] '{query[:40]}' → '{rewritten[:60]}'")
            return rewritten

        return query
    except Exception as e:
        logger.warning(f"[QueryRewrite] 改写失败: {e}，使用原始查询")
        return query


# ============================================================
# 策略2：多路召回
# ============================================================

MULTI_QUERY_PROMPT = """你是查询扩展专家。为同一个问题生成 3 个不同角度的查询语句。

用户问题：{query}

请输出 3 个查询变体，每行一个，只输出查询文本本身，不要编号和解释："""


async def multi_query_rewrite(query: str) -> list[str]:
    """
    生成多个查询变体，多路检索后合并

    "交换机配置" →
      - "交换机 端口配置 VLAN trunk access"
      - "网络设备 配置指南 操作步骤"
      - "交换机 命令行 配置方法"
    """
    try:
        llm = get_chat_model(model="qwen-turbo", temperature=0.3, streaming=False)
        response = await llm.ainvoke(MULTI_QUERY_PROMPT.format(query=query))
        text = response.content.strip() if hasattr(response, "content") else str(response)

        variants = [line.strip() for line in text.split("\n") if line.strip()][:3]
        if variants:
            logger.info(f"[QueryRewrite] 多路召回: {len(variants)} 个变体")
            return variants

        return [query]
    except Exception as e:
        logger.warning(f"[QueryRewrite] 多路召回失败: {e}")
        return [query]


# ============================================================
# 策略3：HyDE（Hypothetical Document Embeddings）
# ============================================================

HYDE_PROMPT = """请根据用户问题，写一段可能出现在技术文档中的回答。
不需要完全准确，只需要风格和术语接近真实文档即可。不超过 100 字。

用户问题：{query}
假设回答："""


async def hyde_rewrite(query: str) -> str:
    """
    HyDE：先让 LLM 生成假设回答，用回答做检索

    原理：用户 query 和文档语言风格不同。LLM 生成的假设回答风格更接近
          真实文档，用假设回答做 Embedding 检索效果更好。

    "CPU飙了" → 假设回答："核心交换机 CPU 使用率持续升高至 95%，
               需要排查进程、检查环路、优化配置"
    → 用这段回答去检索 → 命中真实的故障排查文档
    """
    try:
        llm = get_chat_model(model="qwen-turbo", temperature=0.3, streaming=False)
        response = await llm.ainvoke(HYDE_PROMPT.format(query=query))
        hypothesis = response.content.strip() if hasattr(response, "content") else str(response)

        if hypothesis and 10 < len(hypothesis) < 500:
            logger.info(f"[QueryRewrite] HyDE: '{query[:30]}' → 假设回答 ({len(hypothesis)}字)")
            return hypothesis

        return query
    except Exception as e:
        logger.warning(f"[QueryRewrite] HyDE 失败: {e}")
        return query


# ============================================================
# 统一入口
# ============================================================

async def rewrite_query(
    query: str,
    strategy: str = "llm",  # "llm" | "multi" | "hyde" | "none"
) -> str | list[str]:
    """
    查询改写入统一入口

    Args:
        query: 用户原始查询
        strategy: 改写策略
            - "llm": 单次 LLM 改写（默认，性价比最高）
            - "multi": 多路召回
            - "hyde": 先假设回答再检索
            - "none": 不改写

    Returns:
        改写后的查询字符串（或多路召回时的变体列表）
    """
    if strategy == "none" or not query:
        return query

    if strategy == "llm":
        return await llm_rewrite(query)

    if strategy == "multi":
        return await multi_query_rewrite(query)

    if strategy == "hyde":
        return await hyde_rewrite(query)

    return query
