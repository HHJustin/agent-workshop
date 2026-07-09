"""Agent Workshop — 多模式 Agent 模块

三种 Agent 模式：
    - ReAct（create_agent）        → 适合简单问答
    - Supervisor（主Agent派发）     → 适合多领域路由
    - Plan-Execute-Replan（StateGraph）→ 适合复杂多步诊断
"""

from .intent_router import IntentRouter, intent_router
from .react_agent import ReactAgent
from .supervisor import SupervisorAgent
from .plan_execute import PlanExecuteAgent

__all__ = [
    "ReactAgent",
    "SupervisorAgent",
    "PlanExecuteAgent",
    "IntentRouter",
    "intent_router",
]
