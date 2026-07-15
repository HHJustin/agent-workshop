"""
ToolContext — 统一的工具执行上下文

参考 OpenSquilla 的 ToolContext 设计：
    工具不应该通过零散的 contextvars 获取上下文信息，
    而是通过一个统一的 Context 对象，包含执行所需的全部信息。

Author: 程响
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Literal


CallerKind = Literal["web", "cli", "channel"]


@dataclass
class ToolContext:
    """工具执行上下文 —— 工具调用时传递的完整环境信息"""

    session_id: str = ""
    intent: str = "qa"               # qa / diagnosis / report
    caller: CallerKind = "web"       # 调用来源
    web_search_enabled: bool = True  # 联网开关
    workspace_dir: str = ""          # 工作目录
    metadata: dict = field(default_factory=dict)  # 扩展字段

    @property
    def is_diagnosis(self) -> bool:
        return self.intent == "diagnosis"

    @property
    def is_qa(self) -> bool:
        return self.intent == "qa" or self.intent == "project_intro"

    @property
    def is_report(self) -> bool:
        return self.intent == "report"


# ==================== contextvars (兼容旧 API) ====================

_current_context: contextvars.ContextVar[ToolContext] = contextvars.ContextVar(
    "tool_context", default=ToolContext()
)


def set_current_context(ctx: ToolContext):
    """设置当前工具执行上下文"""
    _current_context.set(ctx)


def get_current_context() -> ToolContext:
    """获取当前工具执行上下文"""
    return _current_context.get()


# ==================== 兼容旧版 set_current_session ====================

def set_current_session(session_id: str):
    """兼容旧版 API：仅设置 session_id"""
    ctx = get_current_context()
    ctx.session_id = session_id
    _current_context.set(ctx)
