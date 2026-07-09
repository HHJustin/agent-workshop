"""
ReAct Agent — LangChain create_agent 模式

适用场景：简单问答、单步工具调用
特点：
    - 开箱即用，create_agent 一把梭
    - LLM 自主思考-行动-观察循环
    - 适合对话式交互

面试考点：
    Q: "ReAct 适用什么场景？为什么这个项目里你三种模式都保留了？"
    A: ReAct 适合简单问答和轻量工具调用——用户问一个配置方法，检索一次知识库即可回答。
       但如果任务需要 4-5 步工具调用（如"分析整个网络的健康状况"），ReAct 容易跑偏。
       所以我同时对比了 Supervisor 和 Plan-Execute-Replan，用 Router 根据意图自动选择。

Author: 程响
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from app.config import config
from app.llm_factory import get_chat_model
from app.logger import logger
from .tools import DEFAULT_TOOLS, set_current_session
from .hooks import HookManager, PresetHooks, HookEvent, HookContext, hook_to_middleware


SYSTEM_PROMPT = """你是一个网络运维智能助手。你可以使用以下工具来帮助用户：

1. retrieve_knowledge — 从知识库检索文档资料（含 BM25+向量+精排混合检索）
2. get_current_time — 获取当前时间
3. search_logs — 查询服务日志
4. query_alerts — 查询系统活动告警
5. web_search — 联网搜索最新信息（知识库没有时使用）
6. mysql_query — 查询 MySQL 数据库（仅 SELECT）
7. prometheus_query — 查询 Prometheus 监控指标
8. send_notification — 诊断完成后发送通知（飞书/钉钉）

## 工作流程（必须遵守！）
- 用户问知识/文档/操作 → 先调 retrieve_knowledge 检索知识库
- 用户要查数据库（带"查询""查一下""统计""有哪些""表""字段""数据"等关键词）→ **必须调 mysql_query，不要用 retrieve_knowledge**
- 用户描述故障/异常 → 先查告警 query_alerts，再查日志 search_logs，最后综合判断
- 知识库没有的信息、需要实时数据（天气/新闻/股价/最新资讯）→ **必须调 web_search 联网搜索**
- 用户问天气但没有说城市 → 从上下文推断或让用户提供，有城市名就调 web_search
- 用户要求通知/推送/发送/汇报 → 诊断完成后**必须调 send_notification** 发送结果
- **重要：不要直接凭记忆回答需要实时数据的问题，必须调工具！**

## 核心约束（违反将导致回答无效）
1. **事实准确**：回答必须严格基于工具查询到的实际数据或知识库检索结果，绝不编造、猜测或添加未获取的信息
2. **诚实原则**：知识库里检索不到相关内容时，直接说"未找到相关资料，无法回答"，不做推测性回答
3. **区分事实和推测**：如果信息不足以得出明确结论，清楚说明"当前证据不足"而非硬给一个答案
4. **数据溯源**：引用工具查询结果时，保留关键字段（告警名称、级别、时间、日志原文等），不泛化
5. **格式要求**：仅输出回答内容本身，使用 Markdown 格式，不附加说明性前缀"""


class ReactAgent:
    """
    ReAct Agent — 最简洁的 Agent 模式

    使用：
        agent = ReactAgent()
        answer = await agent.ainvoke("OSPF 协议怎么配置")
    """

    def rebuild_with_web_search(self, enabled: bool):
        """根据联网开关重建 Agent 的工具集"""
        from .tools import DEFAULT_TOOLS, web_search
        new_tools = list(DEFAULT_TOOLS)
        if not enabled:
            new_tools = [t for t in new_tools if t is not web_search]
        self.tools = new_tools
        self._initialized = False  # 强制下次重建 agent
        logger.info(f"[ReactAgent] 工具集已更新 (web_search={'启用' if enabled else '禁用'})")

    def __init__(self, hooks: HookManager = None):
        self.llm = get_chat_model(temperature=0.1, streaming=True)
        self.tools = list(DEFAULT_TOOLS)
        self.checkpointer = None
        self.agent = None
        self._initialized = False
        # Hook 系统
        self.hooks = hooks or HookManager()
        self._setup_default_hooks()
        logger.info("[ReactAgent] 就绪（Hook 系统已加载）")

    def _setup_default_hooks(self):
        """注册默认中间件：审计 + 安全拦截 + 重试追踪 + 指标收集"""
        # 1. 工具审计追踪
        self.hooks.register(HookEvent.BEFORE_TOOL, PresetHooks.audit_tool_calls())
        self.hooks.register(HookEvent.AFTER_TOOL, PresetHooks.audit_tool_calls())

        # 2. 安全拦截：阻止危险操作
        self.hooks.register(
            HookEvent.BEFORE_TOOL,
            PresetHooks.block_tools(["delete_database", "drop_table", "reboot_server", "shutdown"])
        )

        # 3. 重试追踪：记录每次失败
        self.hooks.register(HookEvent.TOOL_ERROR, PresetHooks.retry_on_error(max_retries=3))

        # 4. 运行时上下文注入（诊断模式时补充提示）
        async def inject_diagnosis_context(ctx: HookContext):
            if ctx.event == HookEvent.BEFORE_MODEL:
                extra = ctx.data.get("extra_context", "")
                ctx.data["extra_context"] = extra + "\n[系统提示] 当前为诊断模式，请严格基于工具查询结果分析，不要推测。"
        self.hooks.register(HookEvent.BEFORE_MODEL, inject_diagnosis_context)

        # 5. 指标收集
        self.hooks.register(HookEvent.AGENT_END, PresetHooks.collect_metrics())

    async def _init(self):
        if self._initialized:
            return
        import os, json
        os.makedirs("data", exist_ok=True)
        self.checkpointer = MemorySaver()

        # 加载 MCP 远程工具（独立进程，动态发现）
        from mcp_tools.mcp_client import mcp_manager
        mcp_tools = await mcp_manager.get_tools()
        all_tools = self.tools + list(mcp_tools)

        self.agent = create_agent(
            model=self.llm, tools=all_tools,
            system_prompt=SYSTEM_PROMPT, checkpointer=self.checkpointer,
        )
        # JSON 文件做持久化备份
        self._data_file = "data/sessions.json"
        self._sessions: dict = self._load_sessions()
        self._initialized = True
        logger.info(f"[ReactAgent] MemorySaver + JSON 持久化就绪 (本地{len(self.tools)}个 + MCP远程{len(mcp_tools)}个)")

    def _load_sessions(self) -> dict:
        import json, os
        if os.path.exists(self._data_file):
            try:
                with open(self._data_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_sessions(self):
        import json
        with open(self._data_file, "w", encoding="utf-8") as f:
            json.dump(self._sessions, f, ensure_ascii=False)

    def _save_messages(self, session_id: str, question: str, answer: str):
        """每次对话后持久化到 JSON 文件"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {"messages": [], "preview": question[:40]}
        self._sessions[session_id]["messages"].append({"role": "user", "content": question})
        self._sessions[session_id]["messages"].append({"role": "assistant", "content": answer})
        self._sessions[session_id]["preview"] = self._sessions[session_id]["messages"][0]["content"][:40]
        self._save_sessions()

    async def get_session_history(self, session_id: str) -> list:
        """获取会话历史消息（优先 JSON 文件）"""
        await self._init()
        # 先查 JSON 文件
        if session_id in self._sessions:
            msgs = self._sessions[session_id].get("messages", [])
            if msgs:
                return msgs
        # 再查 MemorySaver
        config = {"configurable": {"thread_id": session_id}}
        try:
            checkpoint = self.checkpointer.get_tuple(config)
            if not checkpoint:
                return []
            data = checkpoint.checkpoint if hasattr(checkpoint, "checkpoint") else checkpoint[0]
            messages = data.get("channel_values", {}).get("messages", [])
            history = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    continue
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, "content") else str(msg)
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                if content:
                    history.append({"role": role, "content": content})
            return history
        except Exception:
            return []

    async def list_sessions(self) -> list[dict]:
        """列出所有会话"""
        await self._init()
        sessions = []
        for sid, data in self._sessions.items():
            msgs = data.get("messages", [])
            if msgs:
                sessions.append({
                    "id": sid,
                    "preview": data.get("preview", msgs[0]["content"][:40]),
                    "count": len(msgs),
                })
        sessions.sort(key=lambda s: s["id"], reverse=True)
        return sessions

    async def ainvoke(self, query: str, session_id: str = "default") -> str:
        await self._init()
        set_current_session(session_id)  # 会话隔离：注入当前会话ID
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        config_dict = {"configurable": {"thread_id": session_id}}
        result = await self.agent.ainvoke(
            {"messages": messages},
            config=config_dict,
        )
        last_msg = result["messages"][-1]
        return last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    async def astream(self, query: str, session_id: str = "default"):
        await self._init()
        set_current_session(session_id)  # 会话隔离：注入当前会话ID
        await self.hooks.emit(HookEvent.AGENT_START, {"query": query, "session_id": session_id})

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        config_dict = {"configurable": {"thread_id": session_id}}

        async for token, metadata in self.agent.astream(
            {"messages": messages},
            config=config_dict,
            stream_mode="messages",
        ):
            msg_type = type(token).__name__

            # 工具调用：AIMessage 带 tool_calls（跳过，单独处理）
            if hasattr(token, "tool_calls") and token.tool_calls:
                for tc in token.tool_calls:
                    yield {"type": "tool_call", "tool": tc.get("name", "unknown"),
                           "args": str(tc.get("args", {}))[:80]}
                continue

            # 工具结果：ToolMessage
            if msg_type == "ToolMessage":
                yield {"type": "tool_result", "tool": getattr(token, "name", "unknown"),
                       "preview": str(token.content)[:120] if token.content else ""}
                continue

            # LLM 文本（AIMessageChunk 或 AIMessage 都处理）
            if msg_type in ("AIMessageChunk", "AIMessage"):
                text = self._extract_text(token)
                if text:
                    yield {"type": "text", "content": text}

        await self.hooks.emit(HookEvent.AGENT_END, {"session_id": session_id})

    def _extract_text(self, token) -> str:
        """兼容不同版本的 LangChain/DashScope：提取文本内容"""
        # 方式1：content 直接是字符串
        if isinstance(token.content, str) and token.content.strip():
            return token.content

        # 方式2：content_blocks（新版本 LangChain）
        blocks = getattr(token, "content_blocks", None)
        if blocks and isinstance(blocks, list):
            parts = []
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif hasattr(b, "text") and getattr(b, "text", ""):
                    parts.append(b.text)
            return "".join(parts)

        # 方式3：content 是 list
        if isinstance(token.content, list):
            parts = []
            for item in token.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(text)
                elif hasattr(item, "text") and getattr(item, "text", ""):
                    parts.append(item.text)
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)

        # 方式4：additional_kwargs 里的 reasoning_content
        if hasattr(token, "additional_kwargs"):
            ak = token.additional_kwargs
            if isinstance(ak, dict):
                for key in ("content", "text", "reasoning_content"):
                    val = ak.get(key, "")
                    if isinstance(val, str) and val.strip():
                        return val

        return ""


# 全局单例
react_agent = ReactAgent()
