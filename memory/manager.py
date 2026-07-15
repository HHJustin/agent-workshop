"""
MemoryManager — 长期记忆统一入口

流程：
    对话完成 → ImportanceScorer → 评分 ≥ 阈值? → PII脱敏 → 存入 MemoryStore
    用户查询 → 混合检索（FTS5 + 时间衰减） → 注入上下文

API:
    await memory_manager.capture(user_id, content, session_id)  # 自动评分+入库
    memories = memory_manager.retrieve(user_id, query)           # 检索
    memory_manager.delete_all(user_id)                           # Right to be Forgotten

Author: 程响
"""

import asyncio
from app.logger import logger
from .scorer import should_remember
from .store import MemoryStore
from .privacy import mask_pii


class MemoryManager:
    """长期记忆管理器"""

    def __init__(self, store: MemoryStore = None):
        self.store = store or MemoryStore()

    # ─── 捕获 ───

    async def capture(self, user_id: str, content: str, source: str = "") -> dict | None:
        """
        捕获一条消息 → LLM 评分 → 重要则脱敏入库

        Returns:
            入库的 MemoryEntry 或 None（不够重要）
        """
        if len(content) < 10:
            return None

        remember, result = await should_remember(content)
        if not remember:
            logger.debug(f"[Memory] 略过（重要性={result.get('score', 0)}）: {content[:50]}...")
            return None

        # PII 脱敏
        safe_content = mask_pii(content)
        safe_summary = mask_pii(result.get("summary", ""))

        mem_id = self.store.add(
            user_id=user_id,
            content=safe_content,
            summary=safe_summary,
            keywords=result.get("keywords", ""),
            importance=result.get("score", 3),
            source=source,
        )

        logger.info(f"[Memory] 已捕获 #{mem_id} (重要性={result['score']}): {safe_summary[:50]}")
        return {"id": mem_id, "summary": safe_summary, "importance": result["score"]}

    # ─── 检索 ───

    def retrieve(self, user_id: str, query: str, limit: int = 5) -> list[dict]:
        """
        混合检索：FTS5 → 回退 LIKE → 时间衰减排序

        Returns:
            [{"summary": ..., "content": ..., "importance": ..., "age_days": ...}, ...]
        """
        import time

        try:
            memories = self.store.search(user_id, query, limit)
        except Exception:
            memories = self.store.search_fallback(user_id, query, limit)

        now = time.time()
        return [
            {
                "id": m.id,
                "summary": m.summary,
                "content": m.content[:300],
                "keywords": m.keywords,
                "importance": m.importance,
                "age_days": round((now - m.created_at) / 86400, 1),
            }
            for m in memories
        ]

    def retrieve_context(self, user_id: str, query: str, limit: int = 3) -> str:
        """
        检索并格式化为可注入 LLM 的上下文字符串
        """
        memories = self.retrieve(user_id, query, limit)
        if not memories:
            return ""

        lines = ["[长期记忆]"]
        for m in memories:
            lines.append(f"- {m['summary']} ({m['age_days']}天前, 重要性{m['importance']})")
        return "\n".join(lines)

    # ─── 管理 ───

    def delete_all(self, user_id: str) -> int:
        """Right to be Forgotten：删除用户全部记忆"""
        return self.store.soft_delete_all(user_id)

    def stats(self, user_id: str) -> dict:
        return self.store.stats(user_id)


# 全局单例
memory_manager = MemoryManager()
