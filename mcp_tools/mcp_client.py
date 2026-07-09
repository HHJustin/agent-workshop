"""
MCP (Model Context Protocol) 客户端 — 连接外部 MCP Server

面试考点：
    Q: "MCP 和直接调 HTTP API 有什么区别？为什么引入它？"
    A: MCP 是标准化协议（JSON-RPC 2.0），解决了工具发现和调用的标准化问题。
       直接调 HTTP API 需要硬编码每个接口的 URL、参数格式、认证方式。
       MCP 的好处是：① 自动发现——Agent 通过 get_tools() 拿到所有工具的
       name+description+inputSchema，不用预先知道有哪些工具
       ② 进程隔离——MCP Server 独立运行，挂了不影响 Agent 主进程
       ③ 语言无关——MCP Server 可以用 Python/Go/Java 任何语言写

Author: 程响
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import config
from app.logger import logger


# ============================================================
# MCP 工具加载（含重试 + 安全降级）
# ============================================================

async def load_mcp_tools(
    servers: dict[str, dict[str, str]] = None,
    max_retries: int = 2,
) -> list[BaseTool]:
    """
    加载 MCP 工具 — 连接所有 MCP Server，获取可用工具列表

    Args:
        servers: MCP 服务器配置 {name: {transport, url}}
        max_retries: 最大重试次数

    Returns:
        LangChain BaseTool 列表（可直接注册到 Agent）
    """
    if servers is None:
        servers = config.mcp_servers or {}

    if not servers:
        return []

    logger.info(f"[MCP] 正在连接 {len(servers)} 个 MCP Server...")

    for attempt in range(max_retries + 1):
        try:
            client = MultiServerMCPClient(servers)
            tools = await client.get_tools()

            tool_names = [t.name if hasattr(t, "name") else str(t) for t in tools]
            logger.info(f"[MCP] 成功加载 {len(tools)} 个工具: {tool_names}")
            return tools

        except Exception as e:
            if attempt < max_retries:
                delay = 1.0 * (2 ** attempt)
                logger.warning(f"[MCP] 连接失败(尝试{attempt+1}/{max_retries+1}): {e}，{delay}s 后重试")
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[MCP] 重试 {max_retries} 次后仍失败: {e}。"
                    f"降级：仅使用本地工具集，Agent 正常运行。"
                )
                return []


# ============================================================
# MCP 工具接入 Agent
# ============================================================

class MCPToolManager:
    """
    MCP 工具管理器 — 懒加载 + 缓存

    使用：
        mgr = MCPToolManager()
        all_tools = local_tools + await mgr.get_tools()
        agent = create_agent(model=llm, tools=all_tools, ...)
    """

    def __init__(self):
        self._tools: list[BaseTool] = []
        self._loaded = False

    async def get_tools(self) -> list[BaseTool]:
        """获取 MCP 工具（首次调用时加载）"""
        if not self._loaded:
            if config.mcp_enabled:
                self._tools = await load_mcp_tools()
            self._loaded = True
        return self._tools

    def invalidate(self):
        """强制重新加载（MCP Server 重启后调用）"""
        self._loaded = False
        self._tools = []


# 全局实例
mcp_manager = MCPToolManager()
