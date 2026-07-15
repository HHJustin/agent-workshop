"""
Supervisor Agent — 主 Agent 派发子 Agent 模式

适用场景：多领域混合任务，需要根据意图路由到不同专家
特点：
    - 一个 Supervisor 负责分析意图、决定派谁干活
    - 多个子 Agent 各司其职（问答专家、诊断专家）
    - Supervisor 可以中途切换子 Agent

面试考点：
    Q: "多 Agent 怎么协同？Supervisor 和 Router 有什么区别？"
    A: Router 在入口处做一次判断，Supervisor 可以在每轮交互后重新判断。
       如果第1轮问答后发现需要诊断，Supervisor 可以切换到诊断 Agent。
       Router 更适合"一次路由定终身"的场景。

Author: 程响
"""

from typing import Literal

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


# ==================== 子 Agent 定义 ====================

QA_SYSTEM = """你是网络知识问答专家。专注于回答网络技术、设备配置、协议原理等问题。
必须调用 retrieve_knowledge 检索知识库。严格基于检索结果回答，检索不到就说"未找到相关资料"，绝不编造。"""

DIAGNOSIS_SYSTEM = """你是网络故障诊断专家。排查流程：query_alerts→search_logs→综合判断。
严格基于工具返回的实际数据做分析。数据不足时诚实说明"当前证据不足，建议进一步排查"。不编造告警名、日志内容或原因。"""


# ==================== Supervisor 定义 ====================

class SupervisorDecision(BaseModel):
    """Supervisor 的决策输出"""
    next: Literal["qa", "diagnosis", "FINISH"] = Field(
        description="下一步该调用哪个子Agent，或 FINISH 结束"
    )
    reason: str = Field(description="决策理由")


class SupervisorState(TypedDict):
    messages: list  # add_messages 默认行为


SUPERVISOR_PROMPT = """你是 Agent 调度主管。输出 JSON 格式：{"next": "qa|diagnosis|FINISH", "reason": "决策理由"}。

可用的子 Agent：qa（知识问答）、diagnosis（故障诊断）、FINISH（结束）。

决策规则：
1. 用户首次提问 → 问知识/配置/概念 → next="qa"，说故障/告警/异常 → next="diagnosis"
2. 子 Agent 返回结果后 → 信息足够 → next="FINISH"，信息不足 → 选 qa 或 diagnosis 补充
3. 当信息已充分覆盖用户需求时，立即 FINISH
"""


class SupervisorAgent:
    """
    Supervisor Agent — 主 Agent + 子 Agent 协作

    使用：
        agent = SupervisorAgent()
        answer = await agent.ainvoke("交换机 CPU 飙到 95%，怎么排查")
    """

    def __init__(self):
        self.llm = get_chat_model(temperature=0.1)

        # 子 Agent
        self.qa_agent = create_agent(
            model=get_chat_model(temperature=0.1),
            tools=list(DEFAULT_TOOLS),
            system_prompt=QA_SYSTEM,
        )
        self.diagnosis_agent = create_agent(
            model=get_chat_model(temperature=0.1),
            tools=list(DEFAULT_TOOLS),
            system_prompt=DIAGNOSIS_SYSTEM,
        )

        self.graph = self._build_graph()
        logger.info("[SupervisorAgent] 初始化完成 (Supervisor + qa + diagnosis)")

    def _build_graph(self):
        workflow = StateGraph(SupervisorState)
        workflow.add_node("qa", self._call_qa)
        workflow.add_node("diagnosis", self._call_diagnosis)
        workflow.add_node("supervisor", self._call_supervisor)
        workflow.set_entry_point("supervisor")

        workflow.add_conditional_edges("supervisor", lambda s: s.get("next", "FINISH"), {
            "qa": "qa",
            "diagnosis": "diagnosis",
            "FINISH": END,
        })
        workflow.add_edge("qa", "supervisor")
        workflow.add_edge("diagnosis", "supervisor")

        return workflow.compile(checkpointer=MemorySaver())

    async def _call_supervisor(self, state: SupervisorState):
        messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + list(state["messages"])

        # 防循环：超过 5 条消息强制 FINISH
        if len(state["messages"]) > 5:
            logger.info("[Supervisor] 对话已足够长，强制 FINISH")
            return {"next": "FINISH"}

        llm = self.llm.with_structured_output(SupervisorDecision)
        try:
            decision = await llm.ainvoke(messages)
            logger.info(f"[Supervisor] → {decision.next} ({decision.reason[:60]}...)")
            return {"next": decision.next}
        except Exception as e:
            logger.warning(f"[Supervisor] 结构化输出失败: {e}，默认 FINISH")
            return {"next": "FINISH"}

    async def _call_qa(self, state: SupervisorState):
        result = await self.qa_agent.ainvoke({"messages": state["messages"]})
        return {"messages": result["messages"]}

    async def _call_diagnosis(self, state: SupervisorState):
        result = await self.diagnosis_agent.ainvoke({"messages": state["messages"]})
        return {"messages": result["messages"]}

    async def ainvoke(
        self, query: str, session_id: str = "default", intent: str = ""
    ) -> str:
        config = {"configurable": {"thread_id": session_id}}
        # 如果有意图信息，预注入到消息中帮助 Supervisor 做决策
        msgs = [HumanMessage(content=query)]
        if intent:
            msgs.insert(0, SystemMessage(
                content=f"用户意图预判: {intent}（qa=知识问答, diagnosis=故障诊断）"
            ))
        initial = {"messages": msgs}
        result = await self.graph.ainvoke(initial, config=config)
        last_msg = result["messages"][-1]
        return last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    async def astream(
        self, query: str, session_id: str = "default", intent: str = ""
    ):
        config = {"configurable": {"thread_id": session_id}}
        msgs = [HumanMessage(content=query)]
        if intent:
            msgs.insert(0, SystemMessage(
                content=f"用户意图预判: {intent}（qa=知识问答, diagnosis=故障诊断）"
            ))
        initial = {"messages": msgs}

        async for chunk in self.graph.astream(initial, config=config, stream_mode="updates"):
            if not chunk or not hasattr(chunk, "items"):
                continue
            for node_name, node_output in chunk.items():
                if not node_output or not isinstance(node_output, dict):
                    continue
                msgs = node_output.get("messages")
                if msgs and isinstance(msgs, list) and len(msgs) > 0:
                    msg = msgs[-1]
                    if hasattr(msg, "content") and msg.content:
                        yield {"content": msg.content, "node": node_name}


# 全局单例（延迟加载）
_supervisor_agent = None

def get_supervisor_agent():
    global _supervisor_agent
    if _supervisor_agent is None:
        _supervisor_agent = SupervisorAgent()
    return _supervisor_agent
