"""
统一 Checkpoint 管理 — AsyncSqliteSaver 持久化

使用:
    checkpointer = await get_checkpointer("data/checkpoints.db")

Author: 程响
"""

import aiosqlite
import os
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.logger import logger

_checkpointer_cache: dict[str, AsyncSqliteSaver] = {}


async def get_checkpointer(db_path: str = "data/checkpoints.db") -> AsyncSqliteSaver:
    """获取或创建 AsyncSqliteSaver（按 db_path 缓存）"""
    if db_path in _checkpointer_cache:
        return _checkpointer_cache[db_path]

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = await aiosqlite.connect(db_path)
    saver = AsyncSqliteSaver(conn)
    _checkpointer_cache[db_path] = saver
    logger.info(f"[Checkpoint] AsyncSqliteSaver 就绪: {db_path}")
    return saver
