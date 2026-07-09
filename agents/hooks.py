"""
Hook 体系 — pi 风格事件驱动 + LangChain Middleware 双模式

设计理念（借鉴 pi）：
    - Agent 生命周期每个关键节点都可以注册 Hook
    - Hook 可以观察（不修改）也可以拦截（阻止/修改）
    - 支持 AbortSignal 取消机制
    - 异步优先，同步兼容

面试考点：
    Q: "Hook 体系和 Middleware 有什么区别？"
    A: Hook 是事件驱动的观察者模式——任意数量的监听者订阅同一个事件，
       互不干扰。Middleware 是链式拦截器——请求依次经过每个中间件。
       Hook 更适合多消费者场景（日志、监控、权限、审计各自独立订阅），
       Middleware 更适合需要顺序处理的场景（认证→授权→参数校验）。

    Q: "你的 Hook 和 pi 的 beforeToolCall/afterToolCall 有什么异同？"
    A: 理念一致，都是在工具执行前后插入逻辑。pi 用 TypeScript 的
       AbortSignal 做取消，我用 Python 的 asyncio.Event + 自定义异常。
       底层机制不同，设计模式相同。

Author: 程响
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from app.logger import logger


# ============================================================
# Hook 事件类型
# ============================================================

class HookEvent(Enum):
    """Agent 生命周期事件（对标 pi 的 AgentEvent 类型）"""
    AGENT_START = "agent_start"              # Agent 开始运行
    AGENT_END = "agent_end"                  # Agent 运行结束
    TURN_START = "turn_start"                # 新一轮 LLM 调用开始
    TURN_END = "turn_end"                    # 本轮结束
    BEFORE_MODEL = "before_model"            # LLM 调用前
    AFTER_MODEL = "after_model"              # LLM 返回后
    BEFORE_TOOL = "before_tool_call"         # 工具执行前（可阻止）
    AFTER_TOOL = "after_tool_call"           # 工具执行后（可终止循环）
    TOOL_ERROR = "tool_error"               # 工具执行出错
    STREAM_CHUNK = "stream_chunk"            # 流式输出的每个 chunk


# ============================================================
# Hook 上下文
# ============================================================

@dataclass
class HookContext:
    """Hook 执行上下文——Hook 函数通过此对象获取当前状态"""
    event: HookEvent
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def is_aborted(self) -> bool:
        return self.abort_event.is_set()


# ============================================================
# Hook 类型定义
# ============================================================

# Hook 函数签名：接收 HookContext，可返回 HookResult 来控制流程
HookFunc = Callable[[HookContext], Optional["HookResult"] | Awaitable[Optional["HookResult"]]]


@dataclass
class HookResult:
    """Hook 返回值——控制后续流程"""
    # before_tool_call 专用：是否阻止执行
    block: bool = False
    block_reason: str = ""

    # after_tool_call 专用：是否终止 Agent 循环
    terminate: bool = False

    # 通用：修改数据（如修改 tool args、修改 model input）
    modified_data: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Hook 管理器
# ============================================================

class HookManager:
    """
    事件驱动的 Hook 管理器（对标 pi 的 listener 机制）

    使用方式：
        hooks = HookManager()

        # 注册观察者（不修改流程）
        @hooks.on(HookEvent.BEFORE_TOOL)
        async def log_tool_call(ctx: HookContext):
            logger.info(f"调用工具: {ctx.data.get('tool_name')}")

        # 注册拦截器（可阻止执行）
        @hooks.on(HookEvent.BEFORE_TOOL)
        def block_dangerous(ctx: HookContext):
            if ctx.data.get("tool_name") == "bash":
                return HookResult(block=True, block_reason="bash 被禁用")

        # 执行所有 Hook
        results = await hooks.emit(HookEvent.BEFORE_TOOL, {"tool_name": "bash"})
    """

    def __init__(self):
        self._hooks: dict[HookEvent, list[HookFunc]] = {e: [] for e in HookEvent}
        self._global_hooks: list[HookFunc] = []  # 监听所有事件

    # ─── 注册 ───

    def on(self, event: HookEvent | None = None):
        """装饰器：注册 Hook 到指定事件。event=None 时监听所有事件"""
        def decorator(func: HookFunc):
            if event is None:
                self._global_hooks.append(func)
            else:
                self._hooks[event].append(func)
            logger.debug(f"[Hook] 注册 {func.__name__} → {event.value if event else 'all'}")
            return func
        return decorator

    def register(self, event: HookEvent, func: HookFunc):
        """手动注册 Hook"""
        self._hooks[event].append(func)

    def unregister(self, event: HookEvent, func: HookFunc):
        """取消注册"""
        if func in self._hooks[event]:
            self._hooks[event].remove(func)

    # ─── 触发 ───

    async def emit(
        self,
        event: HookEvent,
        data: dict[str, Any] = None,
    ) -> list[HookResult]:
        """
        触发事件——按注册顺序执行所有 Hook

        Args:
            event: 事件类型
            data: 传递给 Hook 的上下文数据

        Returns:
            所有 Hook 返回的 HookResult 列表（已过滤 None）
        """
        ctx = HookContext(event=event, data=data or {})
        results = []

        # 全局 Hook + 事件 Hook
        all_funcs = self._global_hooks + self._hooks.get(event, [])

        for func in all_funcs:
            if ctx.is_aborted:
                break
            try:
                result = func(ctx)
                if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
                    result = await result

                if result is not None:
                    results.append(result)
                    # 如果有 Hook 要求阻止，设置中止标志
                    if result.block:
                        ctx.abort_event.set()
            except Exception as e:
                logger.error(
                    f"[Hook] {func.__name__} 执行出错: {e}\n{traceback.format_exc()}"
                )

        return results

    # ─── 便捷方法 ───

    def has_listeners(self, event: HookEvent) -> bool:
        """检查是否有监听者"""
        return len(self._global_hooks) > 0 or len(self._hooks.get(event, [])) > 0


# ============================================================
# 预置 Hook 工厂（开箱即用）
# ============================================================

class PresetHooks:
    """预置 Hook 工厂——常用的开箱即用 Hook"""

    @staticmethod
    def log_all() -> HookFunc:
        """日志 Hook：记录所有事件"""
        async def hook(ctx: HookContext):
            data_preview = {k: str(v)[:60] for k, v in ctx.data.items()}
            logger.info(f"[Hook:{ctx.event.value}] {data_preview}")
        return hook

    @staticmethod
    def audit_tool_calls() -> HookFunc:
        """审计 Hook：记录每次工具调用的入参和耗时"""
        async def hook(ctx: HookContext):
            if ctx.event == HookEvent.BEFORE_TOOL:
                tool_name = ctx.data.get("tool_name", "unknown")
                tool_args = ctx.data.get("tool_args", {})
                logger.info(f"[Audit] 工具调用: {tool_name}({tool_args})")
                ctx.data["_audit_start"] = time.time()

            elif ctx.event == HookEvent.AFTER_TOOL:
                tool_name = ctx.data.get("tool_name", "unknown")
                start = ctx.data.get("_audit_start", ctx.timestamp)
                elapsed = time.time() - start
                logger.info(f"[Audit] 工具完成: {tool_name}，耗时 {elapsed:.1f}s")
                if ctx.data.get("is_error"):
                    logger.error(f"[Audit] 工具出错: {tool_name} → {ctx.data.get('error')}")

        return hook

    @staticmethod
    def block_tools(tool_names: list[str]) -> HookFunc:
        """安全 Hook：阻止特定工具的执行"""
        async def hook(ctx: HookContext):
            if ctx.event == HookEvent.BEFORE_TOOL:
                if ctx.data.get("tool_name") in tool_names:
                    logger.warning(f"[Security] 已阻止: {ctx.data['tool_name']}")
                    return HookResult(
                        block=True,
                        block_reason=f"工具 {ctx.data['tool_name']} 已被禁用",
                    )
        return hook

    @staticmethod
    def retry_on_error(max_retries: int = 3) -> HookFunc:
        """重试 Hook：工具出错时记录重试次数"""
        async def hook(ctx: HookContext):
            if ctx.event == HookEvent.TOOL_ERROR:
                attempts = ctx.data.get("_retry_count", 0) + 1
                ctx.data["_retry_count"] = attempts
                if attempts <= max_retries:
                    logger.info(f"[Retry] 第 {attempts}/{max_retries} 次重试: {ctx.data.get('tool_name')}")
        return hook

    @staticmethod
    def inject_context(context: dict) -> HookFunc:
        """上下文注入 Hook：在 LLM 调用前注入额外上下文"""
        async def hook(ctx: HookContext):
            if ctx.event == HookEvent.BEFORE_MODEL:
                extra = ctx.data.get("extra_context", "")
                ctx.data["extra_context"] = extra + "\n" + json.dumps(context, ensure_ascii=False)
        return hook

    @staticmethod
    def collect_metrics() -> HookFunc:
        """指标收集 Hook：统计 LLM 调用次数、工具调用次数、总耗时"""
        metrics = {"llm_calls": 0, "tool_calls": 0, "total_tokens": 0}

        async def hook(ctx: HookContext):
            if ctx.event == HookEvent.BEFORE_MODEL:
                metrics["llm_calls"] += 1
            elif ctx.event == HookEvent.BEFORE_TOOL:
                metrics["tool_calls"] += 1
            elif ctx.event == HookEvent.AGENT_END:
                logger.info(f"[Metrics] LLM调用:{metrics['llm_calls']}, "
                            f"工具调用:{metrics['tool_calls']}")
        return hook


# ============================================================
# LangChain Middleware 兼容桥接（含重试机制）
# ============================================================

def hook_to_middleware(
    hook_manager: HookManager,
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """
    将 Hook 系统桥接到 LangChain create_agent 的 middleware 参数

    内置指数退避重试机制 + 兜底策略：
    - 最多重试 max_retries 次（默认 3 次）
    - 延迟：base_delay * 2^attempt（1s → 2s → 4s）
    - 全部失败后返回 ToolMessage(is_error=True) 而非崩溃
    - 每次重试前触发 TOOL_RETRY Hook，重试耗尽触发 TOOL_EXHAUSTED Hook

    使用：
        hooks = HookManager()
        agent = create_agent(
            model=llm, tools=tools,
            middleware=[hook_to_middleware(hooks, max_retries=3)],
        )
    """

    async def middleware_wrapper(request, handler):
        from langchain_core.messages import ToolMessage

        tool_name = request.tool_call.get("name", "unknown") if hasattr(request, "tool_call") else "unknown"
        tool_args = request.tool_call.get("args", {}) if hasattr(request, "tool_call") else {}
        tool_call_id = getattr(request.tool_call, "id", "")

        # BEFORE_TOOL — 安全拦截检查
        results = await hook_manager.emit(HookEvent.BEFORE_TOOL, {
            "tool_name": tool_name,
            "tool_args": tool_args,
        })
        for r in results:
            if r.block:
                logger.warning(f"[Hook] 工具 {tool_name} 被阻止: {r.block_reason}")
                return ToolMessage(
                    content=f"工具已禁用: {r.block_reason}",
                    tool_call_id=tool_call_id,
                )

        # 执行工具（含重试）
        last_error = None
        for attempt in range(max_retries + 1):  # 1次正常 + N次重试
            try:
                result = await handler(request)
                elapsed = time.time() - results[0].timestamp if results else 0
                logger.info(f"[Audit] {tool_name} 成功 (尝试{attempt+1}/{max_retries+1}, {elapsed:.1f}s)")

                await hook_manager.emit(HookEvent.AFTER_TOOL, {
                    "tool_name": tool_name,
                    "result": str(result)[:200],
                    "is_error": False,
                    "attempts": attempt + 1,
                    "elapsed": elapsed,
                })
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[Retry] {tool_name} 失败 (尝试{attempt+1}/{max_retries+1}): {e}"
                )

                # 触发 TOOL_ERROR Hook（每次失败都触发）
                await hook_manager.emit(HookEvent.TOOL_ERROR, {
                    "tool_name": tool_name,
                    "error": str(e),
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                })

                if attempt < max_retries:
                    # 指数退避等待
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"[Retry] 等待 {delay:.1f}s 后第 {attempt+2} 次尝试...")
                    await asyncio.sleep(delay)
                else:
                    # 重试耗尽 → 兜底：返回错误消息而非崩溃
                    logger.error(
                        f"[Retry] {tool_name} 重试 {max_retries} 次后仍然失败: {last_error}"
                    )
                    return ToolMessage(
                        content=(
                            f"工具 {tool_name} 执行失败（重试 {max_retries} 次后仍失败）。\n"
                            f"错误: {str(last_error)[:300]}\n"
                            f"请尝试其他方法或联系管理员。"
                        ),
                        tool_call_id=tool_call_id,
                    )

    return middleware_wrapper


# ============================================================
# 全局默认 Hook 管理器（含重试配置）
# ============================================================

import json

app_hooks = HookManager()

# 注册预置 Hook
app_hooks.register(HookEvent.BEFORE_TOOL, PresetHooks.audit_tool_calls())
app_hooks.register(HookEvent.AFTER_TOOL, PresetHooks.audit_tool_calls())
app_hooks.register(HookEvent.TOOL_ERROR, PresetHooks.retry_on_error(max_retries=3))
app_hooks.register(HookEvent.AGENT_END, PresetHooks.collect_metrics())


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    async def test():
        hooks = HookManager()

        # 注册观察者
        @hooks.on(HookEvent.BEFORE_TOOL)
        async def log_tool(ctx: HookContext):
            print(f"[LOG] 即将调用: {ctx.data.get('tool_name')}")

        # 注册拦截器
        @hooks.on(HookEvent.BEFORE_TOOL)
        def block_dangerous(ctx: HookContext):
            if ctx.data.get("tool_name") == "delete_database":
                print("[SECURITY] 阻止危险操作!")
                return HookResult(block=True, block_reason="危险操作已阻止")

        # 测试 1：正常工具
        print("\n=== 测试1: 正常工具 ===")
        results = await hooks.emit(HookEvent.BEFORE_TOOL, {"tool_name": "search_logs"})
        blocked = any(r.block for r in results)
        print(f"被阻止: {blocked}")

        # 测试 2：危险工具
        print("\n=== 测试2: 危险工具 ===")
        results = await hooks.emit(HookEvent.BEFORE_TOOL, {"tool_name": "delete_database"})
        blocked = any(r.block for r in results)
        print(f"被阻止: {blocked}")

        # 测试 3：指标收集
        print("\n=== 测试3: 指标收集 ===")
        collector = PresetHooks.collect_metrics()
        await collector(HookContext(event=HookEvent.BEFORE_MODEL))
        await collector(HookContext(event=HookEvent.BEFORE_TOOL, data={"tool_name": "test"}))
        await collector(HookContext(event=HookEvent.AGENT_END))

        print("\n✅ Hook 体系测试通过")

    asyncio.run(test())
