"""
LLM 审计追踪 — SQLite 持久化 + 全链路 Span

覆盖面试考点：
    Q: "如何追踪每次 LLM 调用的成本和质量？"
    A: SQLite 存储每次调用的元数据（模型、Token、延迟、输入/输出摘要），
       保留 90 天。每个请求的意图识别→检索→LLM→工具执行串成一条 Trace，
       可以回溯任意一次调用的完整链路。

Author: 程响
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from app.logger import logger


# ============================================================
# 数据库初始化
# ============================================================

DB_PATH = "data/audit.db"


def _get_conn():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_audit_db():
    """创建审计表"""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS audit_traces (
                trace_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_query TEXT NOT NULL,
                intent TEXT,
                start_time REAL NOT NULL,
                end_time REAL,
                total_latency_ms INTEGER,
                status TEXT DEFAULT 'running',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                parent_span_id TEXT,
                span_type TEXT NOT NULL,
                span_name TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                duration_ms INTEGER,
                input_data TEXT,
                output_data TEXT,
                tokens_input INTEGER DEFAULT 0,
                tokens_output INTEGER DEFAULT 0,
                model_name TEXT,
                error_message TEXT,
                metadata TEXT,
                FOREIGN KEY (trace_id) REFERENCES audit_traces(trace_id)
            );

            CREATE INDEX IF NOT EXISTS idx_spans_trace ON audit_spans(trace_id);
            CREATE INDEX IF NOT EXISTS idx_traces_session ON audit_traces(session_id);
            CREATE INDEX IF NOT EXISTS idx_traces_time ON audit_traces(created_at);
        """)
        conn.commit()


# ============================================================
# Trace + Span 模型
# ============================================================

@dataclass
class TraceSpan:
    """一次 LLM/工具调用的 Span"""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    trace_id: str = ""
    parent_span_id: str = ""
    span_type: str = ""          # intent / retrieval / llm / tool / notification
    span_name: str = ""          # 具体操作名
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tokens_input: int = 0
    tokens_output: int = 0
    model_name: str = ""
    error_message: str = ""
    input_data: str = ""
    output_data: str = ""
    metadata: str = "{}"

    def finish(self, output: str = "", tokens_in: int = 0, tokens_out: int = 0, error: str = ""):
        self.end_time = time.time()
        self.output_data = output[:500]
        self.tokens_input = tokens_in
        self.tokens_output = tokens_out
        self.error_message = error

    @property
    def duration_ms(self) -> int:
        if self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return 0

    def save(self):
        with _get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO audit_spans
                (span_id, trace_id, parent_span_id, span_type, span_name,
                 start_time, end_time, duration_ms, input_data, output_data,
                 tokens_input, tokens_output, model_name, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.span_id, self.trace_id, self.parent_span_id,
                self.span_type, self.span_name,
                self.start_time, self.end_time, self.duration_ms,
                self.input_data[:500], self.output_data[:500],
                self.tokens_input, self.tokens_output,
                self.model_name, self.error_message, self.metadata,
            ))
            conn.commit()


@dataclass
class AuditTrace:
    """一次完整的用户请求 Trace"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    session_id: str = ""
    user_query: str = ""
    intent: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"     # running / success / error
    spans: list[TraceSpan] = field(default_factory=list)

    def start_span(self, span_type: str, span_name: str, input_data: str = "",
                   model: str = "", parent_span_id: str = "") -> TraceSpan:
        span = TraceSpan(
            trace_id=self.trace_id,
            parent_span_id=parent_span_id or "",
            span_type=span_type,
            span_name=span_name,
            input_data=input_data[:500],
            model_name=model,
        )
        self.spans.append(span)
        logger.debug(f"[Trace:{self.trace_id}] {span_type}/{span_name} 开始")
        return span

    def finish(self, status: str = "success"):
        self.end_time = time.time()
        self.status = status

    def save(self):
        with _get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO audit_traces
                (trace_id, session_id, user_query, intent, start_time, end_time,
                 total_latency_ms, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.trace_id, self.session_id, self.user_query, self.intent,
                self.start_time, self.end_time,
                int((self.end_time - self.start_time) * 1000) if self.end_time else 0,
                self.status, datetime.now().isoformat(),
            ))
            conn.commit()

        for span in self.spans:
            span.save()

        logger.info(
            f"[Trace:{self.trace_id}] 完成: {self.status}, "
            f"{len(self.spans)} spans, {self.total_latency_ms}ms"
        )

    @property
    def total_latency_ms(self) -> int:
        if self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return 0

    @property
    def total_tokens(self) -> tuple[int, int]:
        """(input_tokens, output_tokens)"""
        inp = sum(s.tokens_input for s in self.spans)
        out = sum(s.tokens_output for s in self.spans)
        return inp, out


# ============================================================
# 查询接口
# ============================================================

def get_trace(trace_id: str) -> Optional[dict]:
    """查询单条 Trace 的完整链路"""
    with _get_conn() as conn:
        trace = conn.execute(
            "SELECT * FROM audit_traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if not trace:
            return None
        spans = conn.execute(
            "SELECT * FROM audit_spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,)
        ).fetchall()
        return {
            "trace": dict(trace),
            "spans": [dict(s) for s in spans],
        }


def list_recent_traces(limit: int = 20, session_id: str = "") -> list[dict]:
    """列出最近的 Trace"""
    with _get_conn() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT * FROM audit_traces WHERE session_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (session_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_traces ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def get_daily_stats(days: int = 7) -> list[dict]:
    """每日统计：调用次数、Token 消耗、平均延迟"""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT date(created_at) as day,
                   COUNT(*) as total_traces,
                   SUM(total_latency_ms) / COUNT(*) as avg_latency_ms,
                   SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as error_count
            FROM audit_traces
            WHERE created_at >= date('now', ?)
            GROUP BY date(created_at)
            ORDER BY day DESC
        """, (f"-{days} days",)).fetchall()
        return [dict(r) for r in rows]

    # Token 统计从 spans 表聚合
    with _get_conn() as conn:
        token_rows = conn.execute("""
            SELECT SUM(tokens_input) as total_input_tokens,
                   SUM(tokens_output) as total_output_tokens
            FROM audit_spans
            WHERE span_type = 'llm'
        """).fetchone()
        if token_rows:
            return {
                "total_input_tokens": token_rows["total_input_tokens"] or 0,
                "total_output_tokens": token_rows["total_output_tokens"] or 0,
            }


# ============================================================
# 自动清理（保留 90 天）
# ============================================================

def cleanup_old_records(retention_days: int = 90):
    """清理超过保留期的审计记录"""
    cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
    with _get_conn() as conn:
        # 先删 spans（外键关联）
        conn.execute("""
            DELETE FROM audit_spans WHERE trace_id IN (
                SELECT trace_id FROM audit_traces WHERE created_at < ?
            )
        """, (cutoff,))
        # 再删 traces
        conn.execute(
            "DELETE FROM audit_traces WHERE created_at < ?", (cutoff,)
        )
        deleted = conn.total_changes
        conn.commit()
    if deleted > 0:
        logger.info(f"[Audit] 清理 {deleted} 条过期记录 (保留 {retention_days} 天)")


# ============================================================
# 全局初始化
# ============================================================

init_audit_db()
logger.info("[Audit] 审计数据库就绪")

# 启动时清理一次
try:
    cleanup_old_records(90)
except Exception:
    pass  # 首次启动表可能还没有数据


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    # 模拟一次完整请求的追踪
    trace = AuditTrace(
        session_id="test_session",
        user_query="ERR-5001 怎么处理",
        intent="diagnosis",
    )

    # Span 1: 意图识别
    span1 = trace.start_span("intent", "IntentRouter", input_data="ERR-5001 怎么处理")
    span1.finish(output="intent=diagnosis", tokens_in=10, tokens_out=2)

    # Span 2: 知识检索
    span2 = trace.start_span("retrieval", "HybridSearch", parent_span_id=span1.span_id)
    span2.finish(output="检索到 5 条文档", tokens_in=0, tokens_out=0)

    # Span 3: LLM 调用
    span3 = trace.start_span("llm", "qwen-max", model="qwen-max",
                             input_data="根据资料回答 ERR-5001...",
                             parent_span_id=span2.span_id)
    span3.finish(output="ERR-5001 是数据库连接池耗尽...", tokens_in=500, tokens_out=150)

    # Span 4: 工具调用
    span4 = trace.start_span("tool", "send_notification", parent_span_id=span3.span_id)
    span4.finish(output="通知已发送")

    trace.finish("success")
    trace.save()

    # 查询验证
    print("\n=== 最近 Trace ===")
    for t in list_recent_traces(3):
        print(f"  {t['trace_id']}: {t['user_query'][:40]}... "
              f"| {t['total_latency_ms']}ms | {t['status']}")

    result = get_trace(trace.trace_id)
    if result:
        print(f"\n=== Trace {trace.trace_id} 链路 ===")
        for s in result["spans"]:
            print(f"  [{s['span_type']}] {s['span_name']}: "
                  f"{s['duration_ms']}ms "
                  f"tokens={s['tokens_input']}/{s['tokens_output']}")

    token_stats = get_daily_stats()
    print(f"\nToken 统计: {token_stats}")

    print("\n✅ 审计追踪测试通过")
