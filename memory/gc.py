"""
Memory GC — 遗忘机制

    1. 低重要性过期清理: 重要性1-2→30天, 3→90天, 4-5→永久
    2. VACUUM 回收空间
    3. 统计日志

使用:
    手动: python -m memory.gc
    定时: APScheduler Cron 每天凌晨 3:00 自动调用

Author: 程响
"""

from .store import MemoryStore
from app.logger import logger


def run_gc():
    """执行完整 GC 循环"""
    store = MemoryStore()

    # 1. 清理过期记忆
    deleted = store.cleanup_all_users()
    if deleted > 0:
        logger.info(f"[MemoryGC] 清理 {deleted} 条过期记忆")
    else:
        logger.debug("[MemoryGC] 无过期记忆")

    # 2. 统计
    stats = store.stats("__all__") if hasattr(store, 'stats') else {}
    logger.info(f"[MemoryGC] 当前状态: {stats}")

    # 3. 回收空间
    if deleted > 10:  # 只有清理量较大时才 VACUUM
        store.vacuum()

    store.close()
    return deleted


if __name__ == "__main__":
    deleted = run_gc()
    print(f"Memory GC complete. Deleted: {deleted}")
