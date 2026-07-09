"""
Boss Agent — Supervisor + PlanExecute 组合模式

架构：
    Supervisor（Boss，始终在线）
    ├── 简单问答 → ReAct Agent（快）
    ├── 复杂诊断 → PlanExecute Agent（深）
    └── 评估后 → 可切换 Sub-Agent 补充信息

面试考点：
    Q: "为什么不用单一 Agent 模式？"
    A: 简单问题走 ReAct（快，token 级流式），复杂问题走 PlanExecute（深，多步规划）。
       Supervisor 根据任务复杂度自动选择，动态切换。
       这比固定 Workflow 灵活，比纯 ReAct 慎重。

Author: 程响
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.config import config
from app.llm_factory import get_chat_model
from app.logger import logger
from .tools import DEFAULT_TOOLS
from .react_agent import SYSTEM_PROMPT as REACT_PROMPT
from .plan_execute import PlanExecuteAgent


# ==================== 状态 ====================

class BossState(TypedDict):
    messages: list


# ==================== Supervisor 决策 ====================

class BossDecision(BaseModel):
    """Boss 决策输出"""
    next: str = Field(default="react", description="react / plan_execute / FINISH")
    action: str = Field(default="react", description="LLM 可能用 action 代替 next")
    reason: str = Field(default="", description="决策理由")

    def get_next(self) -> str:
        val = (self.next or self.action or "react").lower()
        if "finish" in val:
            return "FINISH"
        if "plan" in val or "execute" in val:
            return "plan_execute"
        return "react"


BOSS_PROMPT = """你是 Agent 调度主管（Boss）。根据对话历史决定下一步行动，输出 JSON。

可用的 Sub-Agent：
- react：快速模式（适合简单问答、单步工具调用、知识查询）
- plan_execute：深度模式（适合复杂多步诊断、需要全局规划的故障排查）
- FINISH：信息充足，结束

决策规则：
1. 简单问题（知识问答、单步操作）→ react
2. 复杂诊断（多步排查、需要分析多个数据源）→ plan_execute
3. Sub-Agent 返回后 → 判断信息是否足够
   - 足够 → FINISH
   - 需要补充 → 选择另一个 Sub-Agent
4. 连续 3 次切换后必须 FINISH"""


# ==================== Boss Agent ====================

class BossAgent:
    """
    Boss Agent = Supervisor + ReAct + PlanExecute

    使用：
        boss = BossAgent()
        answer = await boss.ainvoke("核心交换机 CPU 飙到 95%，全面排查")
    """

    def __init__(self):
        self.llm = get_chat_model(temperature=0.1)

        # Sub-Agent 1: ReAct（快速模式）
        self.react_agent = create_agent(
            model=get_chat_model(temperature=0.1),
            tools=list(DEFAULT_TOOLS),
            system_prompt=REACT_PROMPT,
        )

        # Sub-Agent 2: PlanExecute（深度模式）
        self.plan_agent = PlanExecuteAgent()

        self.graph = self._build_graph()
        logger.info("[BossAgent] 初始化完成 (Supervisor + ReAct + PlanExecute)")

    def _build_graph(self):
        workflow = StateGraph(BossState)
        workflow.add_node("react", self._call_react)
        workflow.add_node("plan_execute", self._call_plan_execute)
        workflow.add_node("supervisor", self._call_supervisor)
        workflow.set_entry_point("supervisor")

        workflow.add_conditional_edges("supervisor", lambda s: s.get("next", "FINISH"), {
            "react": "react",
            "plan_execute": "plan_execute",
            "FINISH": END,
        })
        workflow.add_edge("react", "supervisor")
        workflow.add_edge("plan_execute", "supervisor")

        return workflow.compile(checkpointer=MemorySaver())

    async def _call_supervisor(self, state: BossState):
        messages = [SystemMessage(content=BOSS_PROMPT)] + list(state["messages"])
        if len(state["messages"]) > 7:
            logger.info("[Boss] 对话已充分，强制 FINISH")
            return {"next": "FINISH"}

        llm = self.llm.with_structured_output(BossDecision)
        try:
            decision = await llm.ainvoke(messages)
            nxt = decision.get_next()
            logger.info(f"[Boss] → {nxt} ({decision.reason[:60]})")
            return {"next": nxt}
        except Exception as e:
            logger.warning(f"[Boss] 决策失败: {e}，默认 react")
            return {"next": "react"}

    async def _call_react(self, state: BossState):
        """ReAct 子 Agent — 流式收集结果"""
        last_msg = state["messages"][-1] if state["messages"] else None
        query = last_msg.content if hasattr(last_msg, "content") else str(last_msg) if last_msg else ""
        full_content = ""

        async for chunk in self.react_agent.astream(query, "boss_react"):
            if chunk.get("type") == "text" and chunk.get("content"):
                full_content += chunk["content"]
                if hasattr(self, "_stream_queue") and self._stream_queue:
                    await self._stream_queue.put({"content": chunk["content"], "node": "react"})
            elif chunk.get("type") in ("tool_call", "tool_result"):
                if hasattr(self, "_stream_queue") and self._stream_queue:
                    await self._stream_queue.put(chunk)

        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=full_content)]}

    async def _call_plan_execute(self, state: BossState):
        """PlanExecute 子 Agent — 流式输出步骤进度"""
        last_msg = state["messages"][-1] if state["messages"] else None
        query = last_msg.content if hasattr(last_msg, "content") else str(last_msg) if last_msg else ""

        # 流式输出 PlanExecute 的每个步骤
        async for chunk in self.plan_agent.astream(query, "boss_plan"):
            if hasattr(self, "_stream_queue") and self._stream_queue:
                await self._stream_queue.put(chunk)

        # 最终结果
        result = await self.plan_agent.graph.ainvoke(
            {"input": query, "plan": [], "past_steps": [], "response": ""},
            {"configurable": {"thread_id": "boss_plan"}},
        )
        response = result.get("response", "无法生成诊断报告")
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=response)]}

    # ─── 公共接口 ───

    async def ainvoke(self, query: str, session_id: str = "default") -> str:
        config = {"configurable": {"thread_id": session_id}}
        initial = {"messages": [HumanMessage(content=query)]}
        result = await self.graph.ainvoke(initial, config=config)
        last_msg = result["messages"][-1]
        return last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    async def astream(self, query: str, session_id: str = "default"):
        """流式：子 Agent 的每个 token 实时穿透到前端"""
        import asyncio
        self._stream_queue = asyncio.Queue()

        async def run_graph():
            config = {"configurable": {"thread_id": session_id}}
            initial = {"messages": [HumanMessage(content=query)]}
            await self.graph.ainvoke(initial, config=config)
            await self._stream_queue.put(None)  # 结束信号

        task = asyncio.create_task(run_graph())

        while True:
            chunk = await self._stream_queue.get()
            if chunk is None:
                break
            yield chunk

        self._stream_queue = None


# 全局单例
boss_agent = BossAgent()
