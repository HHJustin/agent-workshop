"""
记忆重要性评分 — LLM 判断是否值得存入长期记忆

核心原则：不是所有对话都存。噪音过多 → 检索质量崩盘。
只有 Importance ≥ 3（满分5）的消息才入库。

Author: 程响
"""

from app.llm_factory import get_chat_model
from app.logger import logger

IMPORTANCE_PROMPT = """你是信息重要性评估专家。评估以下对话是否值得存入长期记忆。

评分标准（1-5 分）：
5 — 关键个人信息（姓名、联系方式、地址、重要日期、偏好设置）
4 — 决策/结论/待办事项/项目关键信息
3 — 有用的背景信息、上下文、经验教训
2 — 普通闲聊、问候、简单问答
1 — 纯噪音、无意义消息

输出 JSON 格式：{"score": 3, "summary": "一句话摘要（≤30字）", "keywords": "逗号分隔关键词", "reason": "为什么这个分数"}

对话内容：
{content}"""


async def score_importance(content: str) -> dict:
    """
    评估消息重要性

    Returns:
        {"score": 1-5, "summary": "...", "keywords": "...", "reason": "..."}
        或 {"score": 0, ...} 表示评估失败
    """
    if len(content) < 10:
        return {"score": 0, "summary": "", "keywords": "", "reason": "内容过短"}

    try:
        import json
        llm = get_chat_model(model="qwen-turbo", temperature=0.0, streaming=False)
        response = await llm.ainvoke(IMPORTANCE_PROMPT.format(content=content[:1000]))
        text = response.content if hasattr(response, "content") else str(response)

        # 解析 JSON（LLM 可能输出多余文字）
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            result.setdefault("score", 0)
            result.setdefault("summary", "")
            result.setdefault("keywords", "")
            result.setdefault("reason", "")
            return result
    except Exception as e:
        logger.warning(f"[MemoryScorer] 评估失败: {e}")

    return {"score": 0, "summary": "", "keywords": "", "reason": "评估失败"}


async def should_remember(content: str, threshold: int = 3) -> tuple[bool, dict]:
    """判断是否该记住，返回 (是否记住, 评分详情)"""
    result = await score_importance(content)
    return result.get("score", 0) >= threshold, result
