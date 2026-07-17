"""
长期记忆存储 — SQLite + FTS5 全文搜索 + 时间戳索引

Schema:
    memories(id, user_id, content, summary, keywords, importance, source, created_at, is_deleted)

Author: 程响
"""

from __future__ import annotations

import sqlite3
import time
import os
from dataclasses import dataclass, field

from app.logger import logger


@dataclass
class MemoryEntry:
    id: int = 0
    user_id: str = ""
    content: str = ""
    summary: str = ""
    keywords: str = ""
    importance: int = 0
    source: str = ""       # 来源 session_id
    created_at: float = 0.0
    is_deleted: bool = False


class MemoryStore:
    """SQLite 长期记忆存储（每用户独立分区）"""

    def __init__(self, db_path: str = "data/long_term_memory.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
        logger.info(f"[MemoryStore] 就绪: {db_path}")

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT DEFAULT '',
                keywords TEXT DEFAULT '',
                importance INTEGER DEFAULT 0,
                source TEXT DEFAULT '',
                created_at REAL NOT NULL,
                is_deleted INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at);
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, summary, keywords, content='memories', content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, summary, keywords)
                VALUES (new.id, new.content, new.summary, new.keywords);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, summary, keywords)
                VALUES('delete', old.id, old.content, old.summary, old.keywords);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, summary, keywords)
                VALUES('delete', old.id, old.content, old.summary, old.keywords);
                INSERT INTO memories_fts(rowid, content, summary, keywords)
                VALUES (new.id, new.content, new.summary, new.keywords);
            END;
        """)
        self.conn.commit()

    # ─── CRUD ───

    def add(self, user_id: str, content: str, summary: str, keywords: str,
            importance: int, source: str = "") -> int:
        """添加记忆，返回 id"""
        row = (user_id, content, summary, keywords, importance, source, time.time(), 0)
        cur = self.conn.execute(
            "INSERT INTO memories(user_id,content,summary,keywords,importance,source,created_at,is_deleted) "
            "VALUES(?,?,?,?,?,?,?,?)", row
        )
        self.conn.commit()
        return cur.lastrowid

    def get_by_user(self, user_id: str, limit: int = 50) -> list[MemoryEntry]:
        """获取用户的所有活跃记忆，按重要性和时间排序"""
        rows = self.conn.execute(
            "SELECT id,user_id,content,summary,keywords,importance,source,created_at,is_deleted "
            "FROM memories WHERE user_id=? AND is_deleted=0 "
            "ORDER BY importance DESC, created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        return [MemoryEntry(*r) for r in rows]

    def search(self, user_id: str, query: str, limit: int = 10) -> list[MemoryEntry]:
        """FTS5 关键词搜索 + 时间衰减排序"""
        rows = self.conn.execute(
            "SELECT m.id,m.user_id,m.content,m.summary,m.keywords,m.importance,m.source,m.created_at,m.is_deleted "
            "FROM memories m "
            "INNER JOIN memories_fts fts ON m.id = fts.rowid "
            "WHERE m.user_id=? AND m.is_deleted=0 AND memories_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT ?",
            (user_id, _fts_sanitize(query), limit)
        ).fetchall()
        return [MemoryEntry(*r) for r in rows]

    def search_fallback(self, user_id: str, query: str, limit: int = 10) -> list[MemoryEntry]:
        """FTS 不可用时的 LIKE 回退"""
        like = f"%{query}%"
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE user_id=? AND is_deleted=0 "
            "AND (content LIKE ? OR summary LIKE ? OR keywords LIKE ?) "
            "ORDER BY importance DESC, created_at DESC LIMIT ?",
            (user_id, like, like, like, limit)
        ).fetchall()
        return [MemoryEntry(*r) for r in rows]

    def soft_delete(self, memory_id: int, user_id: str) -> bool:
        """软删除（Right to be Forgotten）"""
        cur = self.conn.execute(
            "UPDATE memories SET is_deleted=1 WHERE id=? AND user_id=?",
            (memory_id, user_id)
        )
        self.conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info(f"[MemoryStore] 已删除 memory#{memory_id}")
        return deleted

    def soft_delete_all(self, user_id: str) -> int:
        """删除用户所有记忆"""
        cur = self.conn.execute(
            "UPDATE memories SET is_deleted=1 WHERE user_id=? AND is_deleted=0",
            (user_id,)
        )
        self.conn.commit()
        logger.info(f"[MemoryStore] 已删除用户 {user_id} 的全部记忆")
        return cur.rowcount

    def search_decay(self, user_id: str, query: str, limit: int = 10,
                     half_life_days: int = 30) -> list[MemoryEntry]:
        """
        FTS5 搜索 + 时间衰减排序

        衰减公式: decay = 1 / (1 + age_days / half_life_days)
        最终分数: importance × decay
        """
        now = time.time()
        try:
            rows = self.conn.execute(
                "SELECT m.id,m.user_id,m.content,m.summary,m.keywords,"
                "m.importance,m.source,m.created_at,m.is_deleted "
                "FROM memories m "
                "INNER JOIN memories_fts fts ON m.id = fts.rowid "
                "WHERE m.user_id=? AND m.is_deleted=0 AND memories_fts MATCH ? "
                "LIMIT 30",  # 多取一些做衰减排序
                (user_id, _fts_sanitize(query))
            ).fetchall()
        except Exception:
            rows = []

        if not rows:
            return self.search_fallback_decay(user_id, query, limit, half_life_days)

        entries = [MemoryEntry(*r) for r in rows]
        scored = [(e, _decay_score(e.importance, e.created_at, now, half_life_days))
                  for e in entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:limit]]

    def search_fallback_decay(self, user_id: str, query: str, limit: int = 10,
                              half_life_days: int = 30) -> list[MemoryEntry]:
        """LIKE 回退版 + 时间衰减"""
        now = time.time()
        like = f"%{query}%"
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE user_id=? AND is_deleted=0 "
            "AND (content LIKE ? OR summary LIKE ? OR keywords LIKE ?) "
            "ORDER BY importance DESC LIMIT 30",
            (user_id, like, like, like)
        ).fetchall()

        entries = [MemoryEntry(*r) for r in rows]
        scored = [(e, _decay_score(e.importance, e.created_at, now, half_life_days))
                  for e in entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:limit]]

    def cleanup_low_importance(self, user_id: str) -> int:
        """
        清理低重要性过期记忆:
          重要性 1-2 → 30 天过期
          重要性 3   → 90 天过期
          重要性 4-5 → 永久保留
        """
        now = time.time()
        deleted = 0

        # 重要性 1-2，超过 30 天
        cursor = self.conn.execute(
            "UPDATE memories SET is_deleted=1 "
            "WHERE user_id=? AND is_deleted=0 AND importance <= 2 "
            "AND (? - created_at) > 2592000",  # 30 天
            (user_id, now)
        )
        deleted += cursor.rowcount

        # 重要性 3，超过 90 天
        cursor = self.conn.execute(
            "UPDATE memories SET is_deleted=1 "
            "WHERE user_id=? AND is_deleted=0 AND importance = 3 "
            "AND (? - created_at) > 7776000",  # 90 天
            (user_id, now)
        )
        deleted += cursor.rowcount

        if deleted > 0:
            self.conn.commit()
            logger.info(f"[MemoryStore] 清理 {deleted} 条过期记忆 (user={user_id})")
        return deleted

    def cleanup_all_users(self) -> int:
        """对所有用户执行低重要性清理"""
        users = self.conn.execute(
            "SELECT DISTINCT user_id FROM memories WHERE is_deleted=0"
        ).fetchall()
        total = 0
        for (user_id,) in users:
            total += self.cleanup_low_importance(user_id)
        return total

    def vacuum(self):
        """回收已删除记录的空间"""
        self.conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('optimize')")
        self.conn.execute("VACUUM")
        logger.info("[MemoryStore] VACUUM 完成")

    def close(self):
        """关闭数据库连接"""
        self.conn.close()

    def stats(self, user_id: str) -> dict:
        """统计信息"""
        total = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id=? AND is_deleted=0", (user_id,)
        ).fetchone()[0]
        avg_imp = self.conn.execute(
            "SELECT AVG(importance) FROM memories WHERE user_id=? AND is_deleted=0", (user_id,)
        ).fetchone()[0] or 0
        return {"total": total, "avg_importance": round(avg_imp, 1)}


def _decay_score(importance: int, created_at: float, now: float, half_life: int = 30) -> float:
    """时间衰减: decay = 1 / (1 + age_days / half_life)，最终 = importance × decay"""
    age_days = max(0, (now - created_at) / 86400)
    decay = 1.0 / (1.0 + age_days / half_life)
    return importance * decay


def _fts_sanitize(query: str) -> str:
    """清理 FTS5 查询字符串，防止语法错误"""
    # 移除 FTS5 特殊字符，用 AND 连接多个词
    import re
    words = re.findall(r'\w+', query)
    return " AND ".join(words[:5]) if words else query.replace("'", "''")
