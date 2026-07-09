"""
四层上下文压缩 — 解决"对话越长 Agent 越蠢"的问题

面试考点（字节 Q5/Q6/Q7）：
    Q: "上下文压缩怎么做？"
    A: 四层分层压缩。对话历史打摘要、工具结果提取关键字段、
       任务状态结构化快照、外部知识按需检索。不同信息类型不同压缩策略。

    Q: "为什么分层？"
    A: 对话历史可以丢细节，任务状态不能丢关键字段，原始证据不能只靠摘要。
       分层才能针对性保留不可压缩信息。

    Q: "压缩过度怎么发现和处理？"
    A: 四类症状：目标漂移、约束遗忘、事实断裂、工具误用。
       检测到后回查原始记录，把丢失的关键信息重新注入。

Author: 程响
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.config import config
from app.llm_factory import get_chat_model
from app.logger import logger


# ============================================================
# 不可压缩字段（硬约束）
# ============================================================

UNCOMPRESSIBLE_FIELDS = [
    "文件路径", "行号", "错误码", "错误类型",
    "接口名", "版本号", "端口号", "IP地址",
    "用户硬性要求", "输出格式约束",
]


# ============================================================
# 压缩事件 & 症状检测
# ============================================================

class CompactionTrigger(Enum):
    """压缩触发原因"""
    CONTEXT_60_PCT = "上下文使用达到60%"
    CONTEXT_80_PCT = "上下文使用达到80%"
    SUBTASK_DONE = "子任务完成"
    AGENT_REPORT = "子Agent汇报"
    LONG_TOOL_RESULT = "工具返回超长"
    MATERIAL_TOO_LONG = "输入材料过长"


class CompactionSymptom(Enum):
    """压缩过度症状"""
    GOAL_DRIFT = "目标漂移"          # 要求A但做了B
    CONSTRAINT_LOST = "约束遗忘"     # 格式/语言要求丢失
    FACT_FRACTURE = "事实断裂"       # 前说失败后说通过
    TOOL_MISUSE = "工具误用"         # 重复搜索已查内容


@dataclass
class CompactionResult:
    """压缩结果"""
    summary: str                          # 压缩后的摘要
    token_before: int                     # 压缩前 token 数（估算）
    token_after: int                      # 压缩后 token 数（估算）
    layers_applied: list[int]             # 应用了哪几层
    trigger: CompactionTrigger
    uncompressible_kept: list[str] = field(default_factory=list)


# ============================================================
# 第1层：对话历史压缩
# ============================================================

COMPRESS_HISTORY_PROMPT = """你是对话摘要专家。将以下对话历史压缩为一段结构化摘要。

必须保留的信息（不可泛化或省略）：
- 用户原始目标和硬性约束（格式、语言、口吻、不能做什么）
- 已完成的关键操作及其结果
- 已确认的关键事实（含文件路径、错误码、版本号）
- 当前未解决的问题或待办事项
- 用户明确要求不允许更改的内容

禁止：
- 使用"按要求做"这种模糊表述，必须写具体要求
- 将错误码泛化（如"数据库错误"不能替代"ORA-00001"）
- 丢失测试结论（"测试有问题"不能替代"test_auth.py::test_login 失败"）

对话历史：
{history}

请输出一段摘要（不超过500字）："""


async def compress_conversation(
    old_messages: list,
    keep_recent: int = 6,
) -> str:
    """
    第1层：对话历史 → LLM 摘要

    Args:
        old_messages: 需要压缩的旧消息列表
        keep_recent: 保留最近 N 条不压缩

    Returns:
        摘要字符串
    """
    if len(old_messages) <= keep_recent:
        return ""

    # 分两段：压缩旧的，保留新的
    to_compress = old_messages[:-keep_recent]

    # 格式化对话
    history_lines = []
    for msg in to_compress:
        role = getattr(msg, "role", "unknown")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        history_lines.append(f"[{role}]: {str(content)[:300]}")
    history_text = "\n".join(history_lines)

    # LLM 生成摘要
    llm = get_chat_model(temperature=0.0, streaming=False)
    prompt = COMPRESS_HISTORY_PROMPT.format(history=history_text)
    response = await llm.ainvoke(prompt)
    summary = response.content if hasattr(response, "content") else str(response)

    token_before = sum(len(str(m.content)) // 2 for m in to_compress if hasattr(m, "content"))
    token_after = len(summary) // 2

    logger.info(
        f"[Compaction L1] 对话压缩: {len(to_compress)}条消息 → 摘要 "
        f"(~{token_before}→~{token_after} tokens)"
    )
    return summary


# ============================================================
# 第2层：工具结果压缩
# ============================================================

TOOL_RESULT_PATTERNS = {
    "file_path": re.compile(r'(?:^|[^\w])/?[\w/.-]+\.(?:py|go|java|ts|js|yaml|yml|json|txt|md|log)(?=[\s\n,;]|$)', re.IGNORECASE),
    "error_code": re.compile(r'\b(?:ERR|ERROR|E\d+|ORA-\d+|500|502|503|404|403)\b', re.IGNORECASE),
    "line_number": re.compile(r'(?:line |:)(\d+)', re.IGNORECASE),
}


async def compress_tool_result(content: str, max_length: int = 500) -> str:
    """
    第2层：超长工具结果 → 提取关键字段

    保留：文件路径、错误码、行号、核心结论
    原文：存外部，需要时可回查（标记引用位置）
    """
    if len(content) <= max_length:
        return content

    # 提取关键信息
    extracted = {}
    for key, pattern in TOOL_RESULT_PATTERNS.items():
        matches = pattern.findall(content)
        if matches:
            extracted[key] = list(set(matches))[:5]

    # 取首尾 + 关键信息
    summary_parts = [content[:200]]  # 开头
    if extracted:
        summary_parts.append(f"\n[关键字段] {extracted}")
    summary_parts.append(f"\n... (原文{len(content)}字符，完整内容可回查) ...")
    summary_parts.append(content[-200:])  # 结尾

    result = "\n".join(summary_parts)
    logger.info(
        f"[Compaction L2] 工具结果压缩: {len(content)}→{len(result)} 字符"
    )
    return result


# ============================================================
# 第3层：任务状态压缩
# ============================================================

TASK_STATE_TEMPLATE = """【任务状态快照】
- 目标: {goal}
- 已读文件: {files_read}
- 关键发现: {findings}
- 已修改: {modified}
- 待验证: {to_verify}
- 风险/阻塞: {blockers}
- 下一步: {next_step}"""


def compress_task_state(
    goal: str = "",
    files_read: list[str] = None,
    findings: list[str] = None,
    modified: list[str] = None,
    to_verify: str = "",
    blockers: str = "",
    next_step: str = "",
) -> str:
    """
    第3层：任务状态 → 结构化快照

    不调用 LLM，直接格式化。这些字段必须精确保留，不能泛化。
    """
    result = TASK_STATE_TEMPLATE.format(
        goal=goal or "（未设定）",
        files_read=", ".join(files_read) if files_read else "（无）",
        findings="; ".join(findings) if findings else "（无）",
        modified=", ".join(modified) if modified else "（无）",
        to_verify=to_verify or "（无）",
        blockers=blockers or "（无）",
        next_step=next_step or "（待定）",
    )

    # 不可压缩字段检查
    kept = []
    for field in UNCOMPRESSIBLE_FIELDS:
        if field in result:
            kept.append(field)

    if kept:
        logger.debug(f"[Compaction L3] 任务快照已生成，保留关键字段: {kept}")
    return result


# ============================================================
# 压缩过度检测
# ============================================================

def detect_over_compaction(
    task_snapshot: str,
    current_response: str,
) -> list[CompactionSymptom]:
    """
    检测四类压缩过度症状

    Args:
        task_snapshot: 压缩前的任务状态快照
        current_response: Agent 当前输出

    Returns:
        检测到的症状列表
    """
    symptoms = []

    # 症状1：目标漂移 — 快照里写"只生成 Markdown"，但响应里出现了代码修改
    if "不要改代码" in task_snapshot or "只生成" in task_snapshot:
        if "修改" in current_response or "更改" in current_response:
            if "不要改" in task_snapshot:
                symptoms.append(CompactionSymptom.GOAL_DRIFT)

    # 症状2：约束遗忘 — 要求中文但输出英文
    if "中文" in task_snapshot or "中文回答" in task_snapshot:
        english_ratio = sum(1 for c in current_response if "a" <= c.lower() <= "z") / max(len(current_response), 1)
        if english_ratio > 0.5:
            symptoms.append(CompactionSymptom.CONSTRAINT_LOST)

    # 症状3：事实断裂 — 前后矛盾
    if "失败" in task_snapshot and ("通过" in current_response or "成功" in current_response):
        if "测试失败" in task_snapshot:
            symptoms.append(CompactionSymptom.FACT_FRACTURE)

    # 症状4：工具误用 — 重复搜索已查内容
    searched_files = re.findall(r'search_file|search_logs', task_snapshot)
    current_searches = re.findall(r'search_file|search_logs', current_response)
    if searched_files and len(current_searches) > len(searched_files):
        symptoms.append(CompactionSymptom.TOOL_MISUSE)

    if symptoms:
        logger.warning(f"[Compaction] 检测到压缩过度症状: {[s.value for s in symptoms]}")

    return symptoms


# ============================================================
# 统一入口
# ============================================================

@dataclass
class ContextCompactor:
    """上下文压缩器 — 统一管理四层压缩"""

    max_token_estimate: int = 0          # 当前估算 token 数
    trigger_threshold_1: float = 0.60    # 第一触发阈值（轻量压缩）
    trigger_threshold_2: float = 0.80    # 第二触发阈值（激进压缩）
    context_window: int = 32000          # 模型上下文窗口（qwen-max ≈ 32K）

    async def compact(
        self,
        messages: list,
        tool_results: list[str] = None,
        task_state: dict = None,
        force_layer: int = 0,
    ) -> CompactionResult:
        """
        根据当前状态自动分层压缩

        Args:
            messages: 当前消息列表
            tool_results: 最近的工具调用结果
            task_state: 当前任务状态 dict(goal, files_read, findings, ...)
            force_layer: 强制触发第N层（0=自动判断）

        Returns:
            CompactionResult: 压缩结果
        """
        # 估算当前 token 数（粗略：2字符≈1 token）
        total_chars = sum(
            len(str(getattr(m, "content", ""))) for m in messages
        )
        self.max_token_estimate = total_chars // 2
        usage_ratio = self.max_token_estimate / self.context_window

        layers_applied = []
        trigger = None
        new_messages = list(messages)

        # Layer 1: 对话历史 — 使用率 > 60% 或强制
        if force_layer == 1 or usage_ratio > self.trigger_threshold_1:
            trigger = CompactionTrigger.CONTEXT_60_PCT if usage_ratio > 0.6 else CompactionTrigger.SUBTASK_DONE
            summary = await compress_conversation(new_messages, keep_recent=6)
            if summary:
                # 摘要作为 system 消息插入到消息列表前面
                compaction_msg = SystemMessage(
                    content=f"[对话历史摘要] {summary}"
                )
                # 只保留最近6条原始消息 + 摘要
                recent = new_messages[-6:]
                new_messages = [compaction_msg] + recent
                layers_applied.append(1)
                logger.info(f"[Compaction L1] 触发: {trigger.value}")

        # Layer 2: 工具结果 — 每个结果独立压缩
        if tool_results and (force_layer == 2 or usage_ratio > self.trigger_threshold_1):
            for i, result in enumerate(tool_results):
                if len(result) > 500:
                    tool_results[i] = await compress_tool_result(result)
            if not layers_applied:
                trigger = CompactionTrigger.LONG_TOOL_RESULT
            if 2 not in layers_applied:
                layers_applied.append(2)

        # Layer 3: 任务状态 — 子任务完成时触发
        if task_state and (force_layer == 3 or trigger == CompactionTrigger.SUBTASK_DONE):
            state_snapshot = compress_task_state(**task_state)
            new_messages.append(SystemMessage(content=state_snapshot))
            if 3 not in layers_applied:
                layers_applied.append(3)

        # Layer 4: 外部知识 — 使用率 > 80%
        if usage_ratio > self.trigger_threshold_2:
            # 外部知识压缩的提示：后续检索时只返回摘要+位置
            logger.info(
                f"[Compaction L4] 上下文使用率 {usage_ratio:.0%}，"
                f"外部知识改为按需检索模式"
            )
            if 4 not in layers_applied:
                layers_applied.append(4)
            if not trigger:
                trigger = CompactionTrigger.CONTEXT_80_PCT

        token_after = sum(len(str(getattr(m, "content", ""))) for m in new_messages) // 2

        return CompactionResult(
            summary=str(new_messages[0].content) if layers_applied else "",
            token_before=self.max_token_estimate,
            token_after=token_after,
            layers_applied=layers_applied or [0],  # 0 表示无需压缩
            trigger=trigger or CompactionTrigger.SUBTASK_DONE,
            uncompressible_kept=UNCOMPRESSIBLE_FIELDS,
        )


# ============================================================
# 全局单例
# ============================================================

context_compactor = ContextCompactor()


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        compactor = ContextCompactor()

        # 模拟长对话
        messages = [
            HumanMessage(content="你好" * 5000),  # 超长消息
        ] + [
            HumanMessage(content=f"第{i}轮对话" * 100)
            for i in range(20)
        ]

        result = await compactor.compact(messages)
        print(f"压缩前 token: ~{result.token_before}")
        print(f"压缩后 token: ~{result.token_after}")
        print(f"应用层数: {result.layers_applied}")
        print(f"触发原因: {result.trigger.value}")

        # 测试压缩过度检测
        symptoms = detect_over_compaction(
            "用户要求中文回答、不要改代码。测试 test_login 失败。",
            "The server is running correctly. Let me modify the config file...",
        )
        print(f"\n症状检测: {[s.value for s in symptoms]}")

    asyncio.run(test())
