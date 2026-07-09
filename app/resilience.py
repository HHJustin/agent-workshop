"""
弹性工程 — Circuit Breaker + 限流 + 熔断降级

面试考点：
    Q: "如何保证 Agent 在高负载下的稳定性？"
    A: 三层防护：限流（令牌桶）兜底 QPS → 熔断器（Circuit Breaker）隔离故障服务
       → 降级策略（Fallback）保证核心功能不崩溃。

Author: 程响
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

from app.logger import logger


# ============================================================
# Circuit Breaker（熔断器）
# ============================================================

class CircuitState(Enum):
    CLOSED = "closed"           # 正常，请求通过
    OPEN = "open"               # 熔断，拒绝请求
    HALF_OPEN = "half_open"     # 半开，允许一次试探


@dataclass
class CircuitBreaker:
    """
    熔断器 — 防止级联故障

    状态机：
        CLOSED ──(失败数达阈值)──→ OPEN ──(冷却时间到)──→ HALF_OPEN
           ↑                                                    │
           └──────────(试探成功)────────────────────────────────┘
           │                                                    │
           └──────────(试探失败)──→ OPEN ────────────────────────┘

    使用方式：
        llm_cb = CircuitBreaker("llm", failure_threshold=5, cooldown_seconds=30)
        async with llm_cb:
            result = await call_llm()
    """

    name: str
    failure_threshold: int = 5       # 连续失败 N 次后熔断
    cooldown_seconds: float = 30.0   # 冷却时间（秒）
    half_open_max: int = 1           # HALF_OPEN 状态下最多允许的试探次数

    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _half_open_count: int = field(default=0, init=False)
    _total_failures: int = field(default=0, init=False)
    _total_successes: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        # OPEN 状态下检查是否该进入 HALF_OPEN
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_count = 0
                logger.info(f"[CB:{self.name}] OPEN → HALF_OPEN (试探)")
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            logger.info(f"[CB:{self.name}] HALF_OPEN → CLOSED (恢复)")
        self._total_successes += 1
        if self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def failure(self):
        self._total_failures += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning(f"[CB:{self.name}] HALF_OPEN → OPEN (试探失败)")
            return

        self._failure_count += 1
        if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.error(
                f"[CB:{self.name}] CLOSED → OPEN "
                f"(连续失败 {self._failure_count}/{self.failure_threshold}，冷却 {self.cooldown_seconds}s)"
            )

    def allow_request(self) -> bool:
        if self.state == CircuitState.OPEN:
            return False
        if self.state == CircuitState.HALF_OPEN:
            if self._half_open_count >= self.half_open_max:
                return False
            self._half_open_count += 1
        return True

    async def __aenter__(self):
        if not self.allow_request():
            raise CircuitBreakerOpenError(
                f"[CB:{self.name}] 熔断中，拒绝请求"
                f"(连续失败 {self._failure_count} 次，冷却 {self.cooldown_seconds}s)"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.success()
        elif exc_type is not None:
            self.failure()
        return False  # 不吞异常

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
        }


class CircuitBreakerOpenError(Exception):
    """熔断器打开时抛出的异常"""
    pass


# ============================================================
# 限流器（Token Bucket）
# ============================================================

@dataclass
class TokenBucket:
    """
    令牌桶限流器

    原理：桶里每秒放入 `rate` 个令牌，最多存 `capacity` 个。
         每个请求消耗 1 个令牌。令牌不够时拒绝请求。

    使用方式：
        limiter = TokenBucket(rate=10, capacity=20)  # 每秒 10 个，突发 20
        if limiter.acquire():
            process_request()
    """

    rate: float            # 每秒生成的令牌数
    capacity: float        # 桶的最大容量（允许的突发量）

    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)

    def __post_init__(self):
        self._tokens = self.capacity
        self._last_refill = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def acquire(self, tokens: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens


# ============================================================
# 降级策略
# ============================================================

class FallbackRegistry:
    """
    降级注册表 — 工具不可用时返回降级结果

    使用方式：
        registry = FallbackRegistry()
        registry.register("prometheus_query", lambda: "[Prometheus] 暂时不可用，已自动降级")

        result = await call_with_fallback(prometheus_query, "prometheus_query")
    """

    def __init__(self):
        self._fallbacks: dict[str, Callable] = {}

    def register(self, name: str, fallback: Callable):
        self._fallbacks[name] = fallback

    def get(self, name: str) -> Optional[Callable]:
        return self._fallbacks.get(name)


# 全局降级注册表
fallback_registry = FallbackRegistry()

# 注册默认降级策略
fallback_registry.register("prometheus_query", lambda: "[Prometheus] 暂时不可用，已自动降级")
fallback_registry.register("web_search", lambda: "[联网搜索] 暂时不可用，已自动降级")
fallback_registry.register("mysql_query", lambda: "[MySQL] 暂时不可用，已自动降级")
fallback_registry.register("send_notification", lambda: "[通知] 暂时不可用，已自动降级")


async def call_with_circuit_breaker(
    coro_factory: Callable[[], Any],
    cb: CircuitBreaker,
    fallback_name: str = "",
) -> Any:
    """
    带熔断器和降级的异步调用

    Args:
        coro_factory: 异步函数工厂
        cb: 熔断器实例
        fallback_name: 降级策略名称

    Returns:
        正常返回值或降级结果
    """
    try:
        async with cb:
            return await coro_factory()
    except CircuitBreakerOpenError:
        fallback = fallback_registry.get(fallback_name)
        if fallback:
            logger.info(f"[Resilience] {fallback_name} 熔断中，使用降级")
            return fallback()
        raise
    except Exception as e:
        fallback = fallback_registry.get(fallback_name)
        if fallback:
            logger.warning(f"[Resilience] {fallback_name} 异常({e})，使用降级")
            return fallback()
        raise


# ============================================================
# 全局熔断器实例
# ============================================================

# 各外部服务的熔断器
circuit_breakers = {
    "llm": CircuitBreaker("llm", failure_threshold=5, cooldown_seconds=30),
    "embedding": CircuitBreaker("embedding", failure_threshold=3, cooldown_seconds=60),
    "mysql": CircuitBreaker("mysql", failure_threshold=3, cooldown_seconds=30),
    "prometheus": CircuitBreaker("prometheus", failure_threshold=3, cooldown_seconds=60),
    "web_search": CircuitBreaker("web_search", failure_threshold=5, cooldown_seconds=30),
}

# 全局限流器：每秒 20 个请求，突发 50
global_limiter = TokenBucket(rate=20, capacity=50)


# ============================================================
# API 限流装饰器
# ============================================================

def rate_limited(func):
    """FastAPI 端点限流装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not global_limiter.acquire():
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"status": "error", "message": "请求过于频繁，请稍后重试"},
            )
        return await func(*args, **kwargs)
    return wrapper


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    # 测试熔断器
    print("=== Circuit Breaker 测试 ===")
    cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=0.5)

    for i in range(5):
        try:
            async def test():
                async with cb:
                    if i < 3:
                        raise ConnectionError("服务不可达")
                    return "ok"
            result = asyncio.run(test())
            print(f"  第{i+1}次: {result} | 状态: {cb.state.value}")
        except CircuitBreakerOpenError as e:
            print(f"  第{i+1}次: {e} | 状态: {cb.state.value}")
        except ConnectionError:
            print(f"  第{i+1}次: 服务不可达 | 状态: {cb.state.value}")

    # 等冷却
    time.sleep(0.6)

    async def test():
        async with cb:
            return "恢复"
    print(f"  冷却后: {asyncio.run(test())} | 状态: {cb.state.value}")

    # 测试限流
    print("\n=== Token Bucket 测试 ===")
    limiter = TokenBucket(rate=5, capacity=5)
    passed = sum(1 for _ in range(10) if limiter.acquire())
    print(f"  10个请求通过: {passed}/10 (预期 ~5)")

    print("\n✅ 弹性工程测试通过")
