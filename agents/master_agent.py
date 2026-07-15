"""
MasterAgent — 统一入口 Agent，内置意图路由 + 工具隔离的子 Agent

架构参考 OpenSquilla TurnRunner 微内核模式：
    用户输入 → IntentRouter（内置）→ 按意图选子 Agent + 工具集 + 执行策略 → 流式输出

子 Agent 真正差异化：
    - qa:       [retrieve_knowledge, web_search, mysql_query, get_current_time]
                流式 ReAct，检索优先
    - diagnosis:[query_alerts, search_logs, prometheus_query, send_notification, MCP]
                PlanExecute，诊断流程
    - report:   [retrieve_knowledge, mysql_query, send_notification, web_search]
                流式 ReAct，数据收集 + 报告

与旧版的区别：
    1. IntentRouter 内置，不再依赖 main.py 做外部路由
    2. 子 Agent 工具集真正隔离——qa 收不到 query_alerts，diagnosis 收不到 retrieve_knowledge
    3. 默认流式输出，不再用 ainvoke
    4. 合并 SupervisorAgent + BossAgent + ReactAgent 为一个入口

Author: 程响
"""

import asyncio

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import config
from app.llm_factory import get_chat_model
from app.logger import logger
from .tools import (
    retrieve_knowledge, get_current_time, search_logs, query_alerts,
    web_search, mysql_query, prometheus_query, send_notification,
)
from .context import ToolContext, set_current_context
from .hooks import HookManager, PresetHooks, HookEvent, HookContext
from .intent_router import intent_router as _router
from .guard import TurnGuard
from mcp_tools.network_server import MCP_NETWORK_TOOLS

# ==================== 工具集定义 ====================

QA_TOOLS = (
    retrieve_knowledge,
    web_search,
    mysql_query,
    get_current_time,
)

DIAGNOSIS_TOOLS = (
    query_alerts,
    search_logs,
    prometheus_query,
    send_notification,
    web_search,
    get_current_time,
) + MCP_NETWORK_TOOLS

REPORT_TOOLS = (
    retrieve_knowledge,
    mysql_query,
    send_notification,
    web_search,
    get_current_time,
)

# ==================== 子 Agent Prompt ====================

QA_SYSTEM = """你是 AI 知识问答专家。专注于知识检索、数据查询和概念解释。

## 工作流程（必须遵守！）
1. **用户问涉及文件、文档、简历、手册、知识的内容 → 第一步必须调 retrieve_knowledge**
2. 知识库检索不到 → 调 web_search 联网搜索
3. 用户要查数据库 → 调 mysql_query（仅 SELECT）
4. 用户问时间相关 → 调 get_current_time

## 核心约束
1. **事实准确**：严格基于工具返回的实际数据回答，绝不编造
2. **诚实原则**：检索不到就说"未找到相关资料"
3. **数据溯源**：引用查询结果时保留关键字段
4. 回答使用 Markdown 格式"""

DIAGNOSIS_SYSTEM = """你是网络故障诊断专家。按标准排障流程工作。

## 排查流程（必须遵守！）
1. 先查告警 → query_alerts
2. 再查日志 → search_logs
3. 需要指标 → prometheus_query
4. 需要外部信息 → web_search
5. 诊断完成 → send_notification 通知运维人员

## 核心约束
1. **严格基于实际数据**：不编造任何告警名、日志内容、监控数值
2. **诚实原则**：证据不足时明确说"当前证据不足，建议进一步排查"
3. **不推测根因**：只描述已确认的事实，不确定的事标注"待确认"
4. 回答使用 Markdown 格式"""

REPORT_SYSTEM = """你是报告生成专家。收集数据并生成结构化报告。

## 工作流程
1. 需要知识库资料 → retrieve_knowledge
2. 需要数据库数据 → mysql_query
3. 需要外部信息 → web_search
4. 汇总后生成 Markdown 格式报告
5. 需要发送 → send_notification

## 核心约束
1. **数据完整**：报告必须基于实际数据，缺数据处标注"待补充"
2. **结构清晰**：使用标题、表格、列表组织内容
3. **不编造**：所有数字必须有来源"""


# ==================== MasterAgent ====================

class MasterAgent:
    """
    统一入口 Agent — 内置路由 + 工具隔离子 Agent

    使用：
        agent = MasterAgent()
        async for chunk in agent.astream("核心交换机 CPU 飙了", "session_1"):
            print(chunk)
    """

    def __init__(self):
        self.llm = get_chat_model(temperature=0.1, streaming=True)

        # 子 Agent：不同的工具集 + Prompt + 执行策略
        self._qa_agent = create_agent(
            model=get_chat_model(temperature=0.1, streaming=True),
            tools=list(QA_TOOLS),
            system_prompt=QA_SYSTEM,
        )
        self._report_agent = create_agent(
            model=get_chat_model(temperature=0.1, streaming=True),
            tools=list(REPORT_TOOLS),
            system_prompt=REPORT_SYSTEM,
        )
        # diagnosis 用 PlanExecute，后面延迟初始化
        self._diagnosis_agent = None

        # Hook 系统
        self.hooks = HookManager()
        self._setup_default_hooks()

        # Session 持久化
        logger.info("[MasterAgent] 就绪 (qa + diagnosis + report)，工具集已隔离")

    def _setup_default_hooks(self):
        self.hooks.register(HookEvent.BEFORE_TOOL, PresetHooks.audit_tool_calls())
        self.hooks.register(HookEvent.AFTER_TOOL, PresetHooks.audit_tool_calls())
        self.hooks.register(HookEvent.BEFORE_TOOL,
            PresetHooks.block_tools(["delete_database", "drop_table", "reboot_server", "shutdown"]))
        self.hooks.register(HookEvent.TOOL_ERROR, PresetHooks.retry_on_error(max_retries=3))
        self.hooks.register(HookEvent.AGENT_END, PresetHooks.collect_metrics())

    async def _get_diagnosis_agent(self):
        """延迟初始化 diagnosis Agent（PlanExecute + AsyncSqliteSaver）"""
        if self._diagnosis_agent is None:
            from .plan_execute import PlanExecuteAgent
            from .checkpoint import get_checkpointer
            ckpt = await get_checkpointer()
            self._diagnosis_agent = PlanExecuteAgent(
                tools=list(DIAGNOSIS_TOOLS), checkpointer=ckpt
            )
        return self._diagnosis_agent

    # ─── 核心路由 ───

    async def astream(self, query: str, session_id: str = "default", intent: str = ""):
        """
        统一的流式入口
        1. 内置 IntentRouter 判断意图
        2. 按意图选子 Agent + 工具集
        3. 流式输出
        """
        set_current_context(ToolContext(session_id=session_id, intent=intent))
        await self.hooks.emit(HookEvent.AGENT_START, {
            "query": query, "session_id": session_id, "intent": intent,
        })

        # Step 1: 内置路由（优先用外部传入的 intent）
        if intent and intent not in ("auto", "plan_execute", "supervisor", "boss", "react"):
            actual_intent = intent  # 外部已判断好的 qa/diagnosis/report
            matched_by = "external"
        else:
            route_result = await _router.route(query)
            actual_intent = route_result.intent
            matched_by = route_result.matched_by
        logger.info(f"[MasterAgent] {query[:30]}... → {actual_intent} ({matched_by})")

        # Step 2: 按意图选子 Agent
        if actual_intent == "diagnosis":
            agent = await self._get_diagnosis_agent()
            logger.info("[MasterAgent] → diagnosis (PlanExecute)")
        elif actual_intent == "report":
            agent = self._report_agent
            logger.info("[MasterAgent] → report (ReAct)")
        else:
            agent = self._qa_agent
            logger.info("[MasterAgent] → qa (ReAct)")

        # Step 3: 流式执行（TurnGuard 多层守护）
        guard = TurnGuard(session_id=session_id,
                          max_seconds=getattr(config, "agent_max_seconds", 120),
                          max_llm_calls=getattr(config, "agent_max_llm_calls", 10),
                          max_tool_errors=getattr(config, "agent_max_tool_errors", 5))

        if actual_intent == "diagnosis":
            async for chunk in agent.astream(query, session_id, intent="diagnosis"):
                budget_err = guard.check_budget()
                if budget_err:
                    yield {"type": "text", "content": f"⏰ {budget_err}"}
                    break
                if chunk.get("type") == "plan":
                    yield {"type": "plan", "steps": chunk["plan"]}
                elif chunk.get("type") == "step":
                    yield {"type": "step", "task": chunk["task"], "result": chunk["result"]}
                elif chunk.get("type") == "text":
                    guard.record_text()
                    yield {"type": "text", "content": chunk["content"]}
        else:
            config_dict = {"configurable": {"thread_id": session_id}}
            async for token, metadata in agent.astream(
                {"messages": [
                    SystemMessage(content=_get_system_prompt(actual_intent)),
                    HumanMessage(content=query),
                ]},
                config=config_dict,
                stream_mode="messages",
            ):
                msg_type = type(token).__name__

                # 工具调用 → 重复检测
                if hasattr(token, "tool_calls") and token.tool_calls:
                    for tc in token.tool_calls:
                        tool_name = tc.get("name", "unknown")
                        args = tc.get("args", {})
                        block = guard.check_repeat(tool_name, args)
                        if block:
                            yield {"type": "text", "content": block}
                            guard.record_tool_error()
                        else:
                            yield {"type": "tool_call", "tool": tool_name,
                                   "args": str(args)[:120]}
                    continue

                # 工具结果 → 错误计数
                if msg_type == "ToolMessage":
                    content = str(token.content) if token.content else ""
                    if "失败" in content or "error" in content.lower():
                        guard.record_tool_error()
                    yield {"type": "tool_result", "tool": getattr(token, "name", "unknown"),
                           "preview": content[:200]}
                    continue

                # 文本 → LLM 调用计数 + 卡死检测
                if msg_type in ("AIMessageChunk", "AIMessage"):
                    guard.record_llm_call()
                    text = _extract_text(token)
                    if text:
                        guard.record_text()
                        yield {"type": "text", "content": text}
                    else:
                        stuck = guard.record_empty()
                        if stuck:
                            yield {"type": "text", "content": stuck}
                            break

                # 预算检查
                budget_err = guard.check_budget()
                if budget_err:
                    yield {"type": "text", "content": f"⏰ {budget_err}"}
                    break

        logger.info(f"[TurnGuard] {guard.elapsed:.1f}s, LLM调用{guard.llm_calls}次, "
                    f"工具错误{guard.tool_errors}次")

        # 成本追踪
        from app.cost_tracker import CostTracker, track_turn
        cost = CostTracker(model=config.llm_model)
        cost.llm_calls = guard.llm_calls
        track_turn(session_id, cost)
        logger.info(f"[Cost] {cost.summary}")

        # 长期记忆捕获
        self._capture_memory(session_id, query)

        await self.hooks.emit(HookEvent.AGENT_END, {"session_id": session_id})

    def _capture_memory(self, user_id: str, content: str):
        """异步触发记忆评分+入库（不阻塞）"""
        from memory.manager import memory_manager
        asyncio.ensure_future(memory_manager.capture(user_id, content, source=user_id))

    # ─── 非流式接口 ───

    async def ainvoke(self, query: str, session_id: str = "default") -> str:
        parts = []
        async for chunk in self.astream(query, session_id):
            if chunk.get("content"):
                parts.append(chunk["content"])
        return "".join(parts)


# ==================== 辅助函数 ====================

def _get_system_prompt(intent: str) -> str:
    """按意图返回子 Agent 的 System Prompt"""
    if intent == "diagnosis":
        return DIAGNOSIS_SYSTEM
    if intent == "report":
        return REPORT_SYSTEM
    return QA_SYSTEM


def _extract_text(token) -> str:
    """兼容不同版本的 LangChain：提取文本内容"""
    if isinstance(token.content, str) and token.content.strip():
        return token.content

    blocks = getattr(token, "content_blocks", None)
    if blocks and isinstance(blocks, list):
        parts = []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif hasattr(b, "text") and getattr(b, "text", ""):
                parts.append(b.text)
        return "".join(parts)

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

    return ""


# 全局单例
master_agent = MasterAgent()
