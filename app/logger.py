"""
日志系统 — Loguru 双通道（控制台 + 文件按天轮转）

面试考点：
    Q: "双通道 Logger 是什么意思？Loguru 多进程安全吗？"
    A: 控制台通道（INFO 级别，调试用）+ 文件通道（DEBUG 级别，排查用）。
       Loguru 默认多进程不安全，需 enqueue=True 开启异步队列。

Author: 程响
"""

import sys

from loguru import logger

from app.config import config


def setup_logger():
    """配置全局日志"""

    # 移除默认 handler
    logger.remove()

    # 通道1：控制台（INFO 以上，带颜色）
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level="DEBUG" if config.debug else "INFO",
        colorize=True,
        backtrace=True,
        diagnose=config.debug,
    )

    # 通道2：文件（DEBUG 以上，按天轮转，自动压缩）— 含 trace_id
    logger.add(
        f"{config.log_dir}/app_{{time:YYYY-MM-DD}}.log",
        rotation="00:00",
        retention=config.log_retention,
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        level="DEBUG",
        format=lambda record: "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {tid:<12} | {name}:{function}:{line} | {message}".format(
            time=record["time"], level=record["level"].name,
            tid=record["extra"].get("trace_id", "-") or "-",
            name=record["name"], function=record["function"],
            line=record["line"], message=record["message"],
        ),
    )

    logger.info(f"[Logger] 日志系统就绪, level={config.log_level}, dir={config.log_dir}")


# 模块导入时自动初始化
setup_logger()
