"""
TurnGuard — 单轮对话的多层守护

1. 上下文溢出    — 估算 token，超过阈值截断早期消息
2. 预算控制      — 单轮时间/LLM调用数/工具错误数上限
3. 重复检测      — 同一工具+参数连续重复 ≥N 次 → 拦截
4. 卡死检测      — 连续 N 轮无文本产出 → 强制终止

参考 OpenSquilla 的 ProgressWatchdog + ToolRunBudgetTracker。

Author: 程响
"""

from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass, field

from app.config import config
from app.logger import logger


@dataclass
class TurnGuard:
    """单轮对话守护器"""

    session_id: str = ""
    start_time: float = 0.0

    # 计数器
    llm_calls: int = 0
    tool_errors: int = 0
    empty_text_rounds: int = 0   # 连续无文本轮数

    # 重复检测
    _tool_call_history: dict[str, int] = field(default_factory=dict)

    # 可配置阈值（从 .env 读取）
    max_seconds: int = 120         # 单轮最长 120 秒
    max_llm_calls: int = 10        # 最多调 LLM 10 次
    max_tool_errors: int = 5       # 最多接受 5 次工具错误
    max_empty_rounds: int = 3      # 连续 3 轮无文本 → 卡死
    max_repeat_calls: int = 3      # 同一工具同参数连续 3 次 → 拦截

    # 上下文溢出
    max_context_chars: int = 30000  # 超过此值截断早期消息

    def __post_init__(self):
        if self.start_time == 0:
            self.start_time = time.time()

    # ─── 1. 上下文溢出 ───

    def trim_messages(self, messages: list) -> list:
        """消息列表超过字符上限时，保留最近的消息"""
        total = sum(len(str(m.content)) if hasattr(m, "content") else 0 for m in messages)
        if total <= self.max_context_chars:
            return messages

        # 从最早的消息开始删，保留 system prompt + 最近几条
        kept = []
        current_len = 0
        # 保留 system message + 从后往前取
        for m in reversed(messages):
            ml = len(str(m.content)) if hasattr(m, "content") else 0
            if current_len + ml > self.max_context_chars and len(kept) >= 2:
                break
            kept.insert(0, m)
            current_len += ml

        dropped = len(messages) - len(kept)
        if dropped > 0:
            logger.warning(f"[TurnGuard] 上下文溢出：{total}→{current_len} 字符，丢弃 {dropped} 条消息")
        return kept

    # ─── 2. 预算控制 ───

    def check_budget(self) -> str | None:
        """检查预算，返回 None=通过，返回错误信息的字符串=超限"""
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            return f"单轮超时 ({elapsed:.0f}s > {self.max_seconds}s)"
        if self.llm_calls >= self.max_llm_calls:
            return f"LLM 调用次数超限 ({self.llm_calls} ≥ {self.max_llm_calls})"
        if self.tool_errors >= self.max_tool_errors:
            return f"工具错误次数超限 ({self.tool_errors} ≥ {self.max_tool_errors})"
        return None

    def record_llm_call(self):
        self.llm_calls += 1

    def record_tool_error(self):
        self.tool_errors += 1

    # ─── 3. 重复检测 ───

    def check_repeat(self, tool_name: str, args: dict) -> str | None:
        """检查工具调用是否重复。返回 None=通过，返回拦截信息=拦截"""
        key = _hash_call(tool_name, args)
        count = self._tool_call_history.get(key, 0) + 1
        self._tool_call_history[key] = count

        if count >= self.max_repeat_calls:
            return (
                f"[系统] 工具 {tool_name} 以相同参数被调用了 {count} 次。"
                f"请换一种方法，不要重复调用。"
            )
        return None

    # ─── 4. 卡死检测 ───

    def record_text(self):
        """有文本产出时重置计数器"""
        self.empty_text_rounds = 0

    def record_empty(self) -> str | None:
        """无文本产出时计数，超过阈值返回终止信息"""
        self.empty_text_rounds += 1
        if self.empty_text_rounds >= self.max_empty_rounds:
            return (
                f"[系统] 已连续 {self.empty_text_rounds} 轮无有效输出，"
                f"任务终止。请基于已有信息给出回答。"
            )
        return None

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


def _hash_call(tool_name: str, args: dict) -> str:
    """计算工具调用的唯一指纹"""
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode()).hexdigest()
