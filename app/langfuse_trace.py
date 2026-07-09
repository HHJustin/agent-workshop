"""
Langfuse 全链路追踪 — Agent 每次调用的完整 Trace

面试考点：
    Q: "如何追踪 Agent 的执行质量？"
    A: 用 Langfuse 做全链路 Trace。每次用户请求自动创建一条 Trace，
       里面的 Span 串联起意图识别→检索→LLM调用→工具执行的完整链路。
       可以在 Dashboard 实时看每次调用的延迟、Token 消耗、工具调用链。

使用方式：
    1. 注册 cloud.langfuse.com → 获取 Public Key + Secret Key
    2. 在 .env 里配置 LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY
    3. 重启服务 → 打开 Langfuse Dashboard 就能看到 Trace

自部署（可选）：
    docker run -p 3000:3000 langfuse/langfuse

Author: 程响
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import config
from app.logger import logger

_langfuse_handler = None


def get_langfuse_handler():
    """获取 Langfuse Callback Handler（懒加载）"""
    global _langfuse_handler

    if _langfuse_handler is not None:
        return _langfuse_handler

    public_key = config.langfuse_public_key
    secret_key = config.langfuse_secret_key

    if not public_key or not secret_key:
        logger.info("[Langfuse] 未配置 API Key，Trace 功能关闭")
        return None

    try:
        from langfuse.langchain import CallbackHandler

        host = config.langfuse_base_url or config.langfuse_host

        # langfuse v4.x: secret_key + host 从环境变量读取
        # 需要设置: LANGFUSE_SECRET_KEY, LANGFUSE_HOST
        import os
        os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
        os.environ.setdefault("LANGFUSE_HOST", host)

        _langfuse_handler = CallbackHandler(
            public_key=public_key,
        )
        logger.info(f"[Langfuse] Trace 已启用 → {config.langfuse_host}")
        return _langfuse_handler
    except ImportError:
        logger.warning("[Langfuse] langfuse 未安装，请执行 pip install langfuse")
        return None
    except Exception as e:
        logger.warning(f"[Langfuse] 初始化失败: {e}")
        return None


def flush():
    """确保所有 Trace 数据发送完毕（优雅关闭时调用）"""
    if _langfuse_handler:
        try:
            _langfuse_handler.flush()
        except Exception:
            pass
