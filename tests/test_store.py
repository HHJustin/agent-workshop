"""MemoryStore 单元测试"""
import os, time
from memory.store import MemoryStore


def _unique_db():
    return f"data/_test_memory_{os.getpid()}_{int(time.time()*1000)}.db"


def teardown_function():
    import glob, time
    time.sleep(0.1)  # 等 SQLite 释放锁
    for f in glob.glob("data/_test_memory_*.db"):
        try:
            os.remove(f)
        except OSError:
            pass


def test_add_and_retrieve():
    s = MemoryStore(db_path=_unique_db())
    try:
        s.add("user1", "我叫程响", "用户名", "名字", 5, "s1")
        memories = s.get_by_user("user1")
        assert len(memories) == 1
        assert memories[0].summary == "用户名"
    finally:
        s.close()


def test_search():
    s = MemoryStore(db_path=_unique_db())
    try:
        s.add("user1", "Python 是最好的语言", "编程偏好", "Python,语言", 4, "s1")
        s.add("user1", "我喜欢喝咖啡", "饮品偏好", "咖啡", 3, "s2")
        results = s.search("user1", "Python")
        assert len(results) >= 1
    finally:
        s.close()


def test_soft_delete():
    s = MemoryStore(db_path=_unique_db())
    try:
        mid = s.add("user1", "secret", "secret", "secret", 1, "s1")
        assert s.soft_delete(mid, "user1")
        assert len(s.get_by_user("user1")) == 0
    finally:
        s.close()


def test_user_isolation():
    s = MemoryStore(db_path=_unique_db())
    try:
        s.add("user1", "data1", "s1", "k1", 3, "s1")
        s.add("user2", "data2", "s2", "k2", 3, "s2")
        assert len(s.get_by_user("user1")) == 1
        assert len(s.get_by_user("user2")) == 1
    finally:
        s.close()


def test_stats():
    s = MemoryStore(db_path=_unique_db())
    try:
        s.add("user1", "a", "a", "a", 3, "s1")
        s.add("user1", "b", "b", "b", 5, "s2")
        stats = s.stats("user1")
        assert stats["total"] == 2
        assert stats["avg_importance"] == 4.0
    finally:
        s.close()
