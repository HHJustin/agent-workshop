"""
后台定时任务 — APScheduler cron

任务：
    1. 健康自检（每 30 分钟）→ 检查 Milvus + LLM 连通性
    2. 日志清理（每天凌晨 3 点）→ 删除超过保留期的日志

Author: 程响
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import config
from app.logger import logger

scheduler = AsyncIOScheduler()


async def health_check():
    """自检：Milvus 连通性"""
    try:
        from retrieval.vector_store import vector_store_manager
        healthy = vector_store_manager.is_healthy
        if healthy:
            logger.info("[Cron] HealthCheck: Milvus OK")
        else:
            logger.warning("[Cron] HealthCheck: Milvus 不可用")
    except Exception as e:
        logger.error(f"[Cron] HealthCheck 失败: {e}")


async def cleanup_old_logs():
    """清理过期日志文件"""
    import os, time, glob
    from pathlib import Path

    log_dir = Path(config.log_dir)
    if not log_dir.exists():
        return

    retention_days = 7
    cutoff = time.time() - retention_days * 86400
    cleaned = 0

    for pattern in ["*.log", "*.zip"]:
        for f in glob.glob(str(log_dir / pattern)):
            if os.path.getmtime(f) < cutoff:
                try:
                    os.remove(f)
                    cleaned += 1
                except OSError:
                    pass

    if cleaned:
        logger.info(f"[Cron] 清理 {cleaned} 个过期日志文件")


def start_scheduler():
    """启动定时任务调度器"""
    if scheduler.running:
        return

    # 每 30 分钟健康自检
    scheduler.add_job(health_check, "interval", minutes=30, id="health_check",
                      replace_existing=True)

    # 每天凌晨 3 点清理日志
    scheduler.add_job(cleanup_old_logs, CronTrigger(hour=3, minute=0),
                      id="log_cleanup", replace_existing=True)

    scheduler.start()
    logger.info("[Scheduler] 定时任务已启动: 健康自检(30min) + 日志清理(每天3:00)")


def stop_scheduler():
    """停止调度器"""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("[Scheduler] 已停止")
