"""TurnGuard 单元测试"""
import time
from agents.guard import TurnGuard


def test_budget_timeout():
    g = TurnGuard(max_seconds=0.001)
    time.sleep(0.01)
    assert g.check_budget() is not None


def test_budget_llm_calls():
    g = TurnGuard(max_llm_calls=3)
    for _ in range(3):
        g.record_llm_call()
    assert g.check_budget() is not None


def test_budget_tool_errors():
    g = TurnGuard(max_tool_errors=2)
    for _ in range(2):
        g.record_tool_error()
    assert g.check_budget() is not None


def test_repeat_detection():
    g = TurnGuard(max_repeat_calls=3)
    assert g.check_repeat("search", {"q": "test"}) is None
    assert g.check_repeat("search", {"q": "test"}) is None
    assert g.check_repeat("search", {"q": "test"}) is not None


def test_repeat_different_args():
    g = TurnGuard(max_repeat_calls=2)
    g.check_repeat("search", {"q": "weather"})
    assert g.check_repeat("search", {"q": "news"}) is None


def test_stuck_detection():
    g = TurnGuard(max_empty_rounds=2)
    g.record_empty()
    assert g.record_empty() is not None


def test_stuck_reset():
    g = TurnGuard(max_empty_rounds=2)
    g.record_empty()
    g.record_text()
    assert g.record_empty() is None


def test_elapsed():
    g = TurnGuard()
    time.sleep(0.1)
    assert g.elapsed > 0
