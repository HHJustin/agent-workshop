"""ToolContext 单元测试"""
from agents.context import ToolContext, set_current_context, get_current_context, set_current_session


def test_default_context():
    ctx = ToolContext()
    assert ctx.session_id == ""
    assert ctx.intent == "qa"
    assert ctx.is_qa


def test_diagnosis_context():
    ctx = ToolContext(session_id="s1", intent="diagnosis")
    assert ctx.is_diagnosis
    assert not ctx.is_qa


def test_set_and_get():
    ctx = ToolContext(session_id="test123", intent="report")
    set_current_context(ctx)
    assert get_current_context().session_id == "test123"
    assert get_current_context().is_report


def test_set_session_compat():
    set_current_session("compat_session")
    assert get_current_context().session_id == "compat_session"
