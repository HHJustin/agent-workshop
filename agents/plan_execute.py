"""
Plan-Execute-Replan Agent — StateGraph 手写模式

适用场景：复杂多步故障诊断
特点：
    - Planner：先检索历史案例 → 生成排查计划
    - Executor：取 plan[0] 逐步执行工具调用
    - Replanner：每步后评估：continue / replan / respond
    - 5 层防循环控制（提示词约束 + 代码强制）

面试考点：
    Q: "为什么用 Plan-Execute-Replan 而不是 ReAct？"
    A: 多步复杂任务（告警→日志→监控→分析→报告）需要全局规划，
       ReAct 边走边看容易偏离方向。P-E-R 有 Planner 做全局规划，
       Replanner 做动态调整，加上 5 层防循环兜底。

Author: 程响
"""

import operator
from textwrap import dedent
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.config import config
from app.llm_factory import get_chat_model
from app.logger import logger
from .tools import DEFAULT_TOOLS


# ==================== 状态定义 ====================

class PlanExecuteState(TypedDict):
    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple], operator.add]
    response: str


# ==================== Pydantic 输出约束 ====================

class Plan(BaseModel):
    steps: list[object] = Field(
        default_factory=list,
        alias="steps",
        description="诊断步骤列表，字符串或 {action, description}",
    )
    plan: list[object] = Field(
        default_factory=list,
        alias="plan",
        description="LLM 可能返回 plan 而不是 steps",
    )

    def get_steps(self) -> list[str]:
        """兼容多种 LLM 输出格式"""
        raw = self.steps or self.plan or []
        result = []
        for s in raw:
            if isinstance(s, str):
                result.append(s)
            elif isinstance(s, dict):
                # 格式1: {action, description}
                # 格式2: {step, action, tool, parameters}
                action = s.get("action") or s.get("tool") or ""
                desc = s.get("description") or s.get("reason") or ""
                params = s.get("parameters", "")
                step_num = s.get("step", "")
                parts = []
                if step_num:
                    parts.append(f"步骤{step_num}")
                if action:
                    parts.append(action)
                if params:
                    parts.append(f"({params})")
                if desc:
                    parts.append(f"— {desc}")
                result.append(" ".join(parts) if parts else str(s))
            else:
                result.append(str(s))
        return result


class Response(BaseModel):
    response: str = Field(default="", description="最终诊断报告")
    report: str = Field(default="", description="LLM 可能返回 report 字段")

    def get_response(self) -> str:
        r = self.response or self.report or ""
        if isinstance(r, dict):
            return str(r)
        return r


class Act(BaseModel):
    action: str = Field(default="continue", description="continue / replan / respond")
    decision: str = Field(default="continue", alias="decision", description="LLM 可能用 decision 代替 action")
    new_steps: list[object] = Field(default_factory=list, description="新步骤（仅 replan 时填写）")

    def get_action(self) -> str:
        val = self.action or self.decision or "continue"
        # 统一：respond > replan > continue
        if "respond" in val.lower() or "finish" in val.lower():
            return "respond"
        if "replan" in val.lower():
            return "replan"
        return "continue"

    def get_new_steps(self) -> list[str]:
        result = []
        for s in self.new_steps:
            if isinstance(s, str):
                result.append(s)
            elif isinstance(s, dict):
                result.append(str(s))
            else:
                result.append(str(s))
        return result


# ==================== 提示词（按意图动态切换） ====================

PROMPTS = {
    "diagnosis": {
        "planner": ChatPromptTemplate.from_messages([
            ("system", dedent("""\
            你是网络故障诊断规划专家。根据用户描述的故障和可用工具，制定排查计划。
            输出 JSON 格式的计划。

            可用工具：{tools_description}

            {experience}

            规则：
            - 每步必须具体：明确使用哪个工具、参数是什么
            - 步骤有逻辑依赖关系：先查告警→再查日志→分析→总结
            - 3-5 步即可，不要过度拆分
            """)),
            ("user", "故障描述：{input}"),
        ]),
        "executor": dedent("""\
        你是故障排查执行专家。根据当前步骤描述，选择合适的工具执行。
        只执行当前这一个步骤，不要考虑其他步骤。如果工具调用失败，说明原因。
        严格基于工具返回的实际数据，不编造任何数值、告警名或日志内容。
        """),
        "replanner": ChatPromptTemplate.from_messages([
            ("system", dedent("""\
            你是诊断评估专家。根据已执行步骤判断下一步行动，输出 JSON 格式。

            三种决策（优先级从高到低）：
            1. respond — 信息已经充足，立即生成最终诊断报告（优先选这个！）
            2. continue — 当前计划合理，继续执行下一步
            3. replan — 计划有严重问题，需要调整（谨慎使用）

            标准：
            - 已有 ≥3 步且获取了关键信息 → 优先 respond
            - 已有 ≥5 步 → 必须 respond
            - 当前信息能回答用户问题时 → 立即 respond，不等所有步骤完成
            """)),
            ("user", "原始问题：{input}\n已执行步骤：{past_steps}\n剩余计划：{plan}"),
        ]),
        "response": ChatPromptTemplate.from_messages([
            ("system", "基于已执行的诊断步骤生成最终报告（JSON 格式）。使用 Markdown 语法。严格基于实际数据，不编造任何信息。数据不足处明确标注'待确认'。"),
            ("user", "原始问题：{input}\n执行记录：{past_steps}"),
        ]),
    },
    "qa": {
        "planner": ChatPromptTemplate.from_messages([
            ("system", dedent("""\
            你是知识问答规划专家。根据用户问题，制定信息收集计划。
            输出 JSON 格式的计划。

            可用工具：{tools_description}

            规则：
            - 第一步优先用 retrieve_knowledge 从知识库检索
            - 知识库没有的资料用 web_search 联网搜索
            - 需要实时数据（天气/新闻）直接 web_search
            - 最后一步综合信息生成回答
            - 2-4 步即可
            """)),
            ("user", "用户问题：{input}"),
        ]),
        "executor": dedent("""\
        你是信息获取执行专家。根据当前步骤描述，选择合适的工具执行。
        只执行当前这一个步骤。检索时使用具体的查询关键词。
        严格基于工具返回的实际内容，不编造任何信息。
        """),
        "replanner": ChatPromptTemplate.from_messages([
            ("system", dedent("""\
            你是信息完整性评估专家。根据已执行步骤判断下一步，输出 JSON 格式。

            三种决策：
            1. respond — 已收集足够信息回答用户问题（优先选这个！）
            2. continue — 当前计划合理，继续执行下一步
            3. replan — 信息不足，调整检索策略

            标准：
            - 已有 ≥2 步且获取了关键信息 → 优先 respond
            - 已有 ≥4 步 → 必须 respond
            - 知识库 + 联网搜索都返回结果时 → 立即 respond
            """)),
            ("user", "原始问题：{input}\n已执行步骤：{past_steps}\n剩余计划：{plan}"),
        ]),
        "response": ChatPromptTemplate.from_messages([
            ("system", "基于收集到的信息生成最终回答（JSON 格式）。使用 Markdown 语法，综合知识库和联网搜索结果，给出准确完整的回答。信息不足处标注'未找到相关资料'。"),
            ("user", "原始问题：{input}\n执行记录：{past_steps}"),
        ]),
    },
    "report": {
        "planner": ChatPromptTemplate.from_messages([
            ("system", dedent("""\
            你是报告生成规划专家。根据用户需求，制定数据收集和报告生成计划。
            输出 JSON 格式的计划。

            可用工具：{tools_description}

            规则：
            - 了解需求：先确定报告的数据来源和格式
            - 收集数据：检索、查询、统计
            - 生成报告：最终汇总输出
            - 3-5 步即可
            """)),
            ("user", "报告需求：{input}"),
        ]),
        "executor": dedent("""\
        你是报告数据采集执行专家。根据当前步骤描述，选择合适的工具执行。
        只执行当前这一步。严格基于工具返回的实际数据。
        """),
        "replanner": ChatPromptTemplate.from_messages([
            ("system", dedent("""\
            你是报告完整性评估专家。根据已执行步骤判断下一步，输出 JSON 格式。

            三种决策：
            1. respond — 数据已充足，立即生成报告
            2. continue — 继续收集数据
            3. replan — 调整数据收集策略

            标准：
            - 已有 ≥3 步且数据充分 → 优先 respond
            - 已有 ≥5 步 → 必须 respond
            """)),
            ("user", "原始需求：{input}\n已执行步骤：{past_steps}\n剩余计划：{plan}"),
        ]),
        "response": ChatPromptTemplate.from_messages([
            ("system", "基于收集到的数据生成最终报告（JSON 格式）。使用 Markdown 语法，结构清晰。数据不足处注明。"),
            ("user", "原始需求：{input}\n执行记录：{past_steps}"),
        ]),
    },
}

# 默认兜底用 qa
DEFAULT_INTENT = "qa"


def _get_prompts(intent: str) -> dict:
    """获取指定意图的提示词，fallback 到 qa"""
    return PROMPTS.get(intent, PROMPTS[DEFAULT_INTENT])


# ==================== 格式化辅助 ====================

def _format_tools(tools) -> str:
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)


def _format_past_steps(steps: list[tuple]) -> str:
    if not steps:
        return "（无）"
    return "\n".join(
        f"步骤{i}: {s}\n结果: {r[:300]}" for i, (s, r) in enumerate(steps, 1)
    )


# ==================== Agent 实现 ====================

class PlanExecuteAgent:
    """
    Plan-Execute-Replan Agent

    使用：
        agent = PlanExecuteAgent()
        answer = await agent.ainvoke("核心交换机 CPU 飙到 95%，排查")
    """

    def __init__(self):
        self.llm = get_chat_model(temperature=0.0, streaming=False)
        self.tools = list(DEFAULT_TOOLS)
        self.intent = DEFAULT_INTENT
        self.graph = self._build_graph()
        logger.info("[PlanExecuteAgent] 初始化完成 (Planner + Executor + Replanner)")

    def _build_graph(self):
        workflow = StateGraph(PlanExecuteState)
        workflow.add_node("planner", self._planner)
        workflow.add_node("executor", self._executor)
        workflow.add_node("replanner", self._replanner)
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "replanner")
        workflow.add_conditional_edges("replanner", self._should_continue, {
            "executor": "executor",
            END: END,
        })
        import os; os.makedirs("data", exist_ok=True)
        return workflow.compile(checkpointer=SqliteSaver.from_conn_string("data/checkpoints.db"))

    # ─── Planner ───

    async def _planner(self, state: PlanExecuteState) -> dict:
        logger.info(f"[Planner] 制定{self.intent}计划...")

        # 检索历史案例作为参考经验（仅 diagnosis 需要）
        experience = ""
        if self.intent == "diagnosis":
            try:
                from retrieval.vector_store import vector_store_manager
                docs = vector_store_manager.similarity_search(state["input"], k=2)
                if docs:
                    experience = "相关历史案例：\n" + "\n".join(
                        f"- {d.page_content[:200]}" for d in docs
                    )
            except Exception as e:
                logger.warning(f"[Planner] 检索经验失败: {e}")

        prompts = _get_prompts(self.intent)
        chain = prompts["planner"] | self.llm.with_structured_output(Plan)
        result = await chain.ainvoke({
            "input": state["input"],
            "tools_description": _format_tools(self.tools),
            "experience": experience,
        })

        plan = result.get_steps() if isinstance(result, Plan) else []
        if not plan:
            # fallback: 尝试从 result 直接取
            if hasattr(result, 'model_dump'):
                raw = result.model_dump()
                plan = raw.get("steps") or raw.get("plan") or []
                if plan and isinstance(plan[0], dict):
                    plan = [s.get("action", str(s)) for s in plan]
        if not plan:
            logger.warning(f"[Planner] LLM 返回空计划！result={result}")
            # qa 兜底：至少检索一次知识库
            if self.intent == "qa":
                plan = ["从知识库检索相关信息并综合回答"]
            elif self.intent == "report":
                plan = ["收集必要数据并生成报告"]
            else:
                plan = ["综合分析当前信息并给出结论"]
        logger.info(f"[Planner] 计划: {len(plan)} 步")
        for i, s in enumerate(plan, 1):
            logger.info(f"  步骤{i}: {s}")

        return {"plan": plan}

    # ─── Executor ───

    async def _executor(self, state: PlanExecuteState) -> dict:
        plan = state.get("plan", [])
        if not plan:
            return {}

        task = plan[0]
        logger.info(f"[Executor] 执行: {task}")

        try:
            from langchain.agents import create_agent
            executor_prompt = _get_prompts(self.intent)["executor"]
            agent = create_agent(
                model=get_chat_model(temperature=0.0),
                tools=self.tools,
                system_prompt=executor_prompt,
            )
            messages = [
                SystemMessage(content=executor_prompt),
                HumanMessage(content=f"执行以下步骤: {task}"),
            ]
            result = await agent.ainvoke({"messages": messages})
            last_msg = result["messages"][-1]
            task_result = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        except Exception as e:
            task_result = f"执行失败: {e}"

        logger.info(f"[Executor] 完成, 结果长度: {len(task_result)}")
        return {
            "plan": plan[1:],
            "past_steps": [(task, task_result)],
        }

    # ─── Replanner ───

    async def _replanner(self, state: PlanExecuteState) -> dict:
        plan = state.get("plan", [])
        past = state.get("past_steps", [])

        # 第3层：≥8 步强制 respond
        if len(past) >= config.agent_max_steps:
            logger.warning(f"[Replanner] 已达最大步数 {config.agent_max_steps}，强制 respond")
            return await self._generate_response(state)

        # 没有剩余计划 → 生成报告
        if not plan:
            logger.info("[Replanner] 计划执行完毕，生成报告")
            return await self._generate_response(state)

        # LLM 决策
        prompts = _get_prompts(self.intent)
        chain = prompts["replanner"] | self.llm.with_structured_output(Act)
        result = await chain.ainvoke({
            "input": state["input"],
            "past_steps": _format_past_steps(past),
            "plan": "\n".join(f"- {s}" for s in plan),
        })
        action = result.get_action() if isinstance(result, Act) else result.get("action", "continue")

        # 第4层：≥5 步禁止 replan
        if action == "replan" and len(past) >= 5:
            logger.warning("[Replanner] 已执行 ≥5 步，禁止 replan，强制 respond")
            return await self._generate_response(state)

        # 第5层：新步骤截断
        if action == "replan":
            new_steps = result.get_new_steps() if isinstance(result, Act) else result.get("new_steps", [])
            if len(new_steps) > len(plan):
                new_steps = new_steps[:len(plan)]
                logger.warning(f"[Replanner] 新步骤数超限，截断至 {len(new_steps)}")
            logger.info(f"[Replanner] replan → {len(new_steps)} 个新步骤")
            return {"plan": new_steps}

        if action == "respond":
            logger.info("[Replanner] respond → 生成最终报告")
            return await self._generate_response(state)

        logger.info("[Replanner] continue → 继续执行")
        return {}

    # ─── 条件边判断 ───

    def _should_continue(self, state: PlanExecuteState) -> str:
        if state.get("response"):
            return END
        if state.get("plan"):
            return "executor"
        return END

    # ─── 生成最终报告 ───

    async def _generate_response(self, state: PlanExecuteState) -> dict:
        prompts = _get_prompts(self.intent)
        chain = prompts["response"] | self.llm.with_structured_output(Response)
        result = await chain.ainvoke({
            "input": state["input"],
            "past_steps": _format_past_steps(state.get("past_steps", [])),
        })
        response = result.get_response() if isinstance(result, Response) else result.get("response", result.get("report", ""))
        return {"response": response}

    # ─── 公共接口 ───

    async def ainvoke(self, query: str, session_id: str = "default", intent: str = "qa") -> str:
        self.intent = intent
        config = {"configurable": {"thread_id": session_id}}
        initial: PlanExecuteState = {"input": query, "plan": [], "past_steps": [], "response": ""}
        result = await self.graph.ainvoke(initial, config=config)
        return result.get("response", "无法生成诊断报告")

    async def astream(self, query: str, session_id: str = "default", intent: str = "qa"):
        self.intent = intent  # 动态切换 Prompt
        config = {"configurable": {"thread_id": session_id}}
        initial: PlanExecuteState = {"input": query, "plan": [], "past_steps": [], "response": ""}

        async for chunk in self.graph.astream(initial, config=config, stream_mode="updates"):
            if not chunk or not hasattr(chunk, "items"):
                continue
            for node_name, node_output in chunk.items():
                if not node_output or not isinstance(node_output, dict):
                    continue
                if node_name == "planner":
                    yield {"type": "plan", "plan": node_output.get("plan", [])}
                elif node_name == "executor":
                    past = node_output.get("past_steps", [])
                    if past:
                        yield {"type": "step", "task": past[-1][0], "result": past[-1][1][:100]}
                elif node_name == "replanner":
                    resp = node_output.get("response", "")
                    if resp:
                        yield {"type": "report", "content": resp}


# 全局单例
plan_execute_agent = PlanExecuteAgent()
