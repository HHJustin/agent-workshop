"""
增量更新 + 版本管理 — doc_id + chunk_id + version 三级管理

面试考点：
    Q: "文档更新了怎么处理？全量重建还是增量更新？"
    A: 增量更新。用 doc_id + chunk_id + version 三级 ID 做 hash diff。
       相同 chunk 不动，新增写入，删除软删除（保留审计），变化重新 embedding。
       旧版本标记 archived 不物理删除——出问题可回滚。

    Q: "为什么要软删除而不是硬删除？"
    A: 生产环境不能直接删数据。软删除后旧版本不可检索但保留在库里，
       万一新版本索引有问题可以快速回滚。同时保留审计痕迹。

Author: 程响
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.logger import logger


# ============================================================
# 索引元数据
# ============================================================

@dataclass
class DocMeta:
    """文档元数据"""
    doc_id: str            # 文档唯一标识（基于文件名+路径）
    version: str           # 版本号（基于内容 hash + 时间戳）
    file_path: str         # 文件路径
    file_hash: str         # 文件级别 MD5
    chunk_count: int = 0   # 分片数量
    status: str = "active" # active / archived
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ChunkMeta:
    """分片元数据"""
    chunk_id: str          # chunk_id = doc_id + section + chunk_index + content_hash[:8]
    doc_id: str
    version: str
    content_hash: str      # 内容 MD5
    section: str           # 所属章节（h1>h2）
    chunk_index: int       # 在文档中的序号
    status: str = "active" # active / deleted


@dataclass
class IndexDiff:
    """新旧版本差异"""
    doc_id: str
    old_version: str
    new_version: str
    unchanged: list[str] = field(default_factory=list)   # 不变的 chunk_id
    added: list[str] = field(default_factory=list)        # 新增的 chunk_id
    deleted: list[str] = field(default_factory=list)      # 删除的 chunk_id
    changed: list[str] = field(default_factory=list)      # 内容变化的 chunk_id
    total_old: int = 0
    total_new: int = 0

    @property
    def is_identical(self) -> bool:
        return len(self.added) == 0 and len(self.deleted) == 0 and len(self.changed) == 0

    @property
    def summary(self) -> str:
        return (f"{self.old_version[:8]}→{self.new_version[:8]}: "
                f"不变={len(self.unchanged)}, 新增={len(self.added)}, "
                f"删除={len(self.deleted)}, 变化={len(self.changed)}")


# ============================================================
# 元数据存储（JSON 文件）
# ============================================================

class IndexRegistry:
    """
    索引注册表 — 维护文档和分片的元数据

    存储位置：data/index_registry.json
    结构：
    {
        "docs": {
            "doc_id": {
                "versions": {
                    "version_hash": DocMeta,
                    ...
                },
                "active_version": "version_hash"
            }
        },
        "chunks": {
            "chunk_id": ChunkMeta
        }
    }
    """

    def __init__(self, registry_path: str = "data/index_registry.json"):
        self.path = registry_path
        self.data: dict = {"docs": {}, "chunks": {}}
        self._load()

    def _load(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"docs": {}, "chunks": {}}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # ─── 文档 ───

    def get_doc(self, doc_id: str) -> Optional[dict]:
        return self.data["docs"].get(doc_id)

    def get_active_version(self, doc_id: str) -> Optional[str]:
        doc = self.get_doc(doc_id)
        return doc.get("active_version") if doc else None

    def register_doc(self, meta: DocMeta):
        if meta.doc_id not in self.data["docs"]:
            self.data["docs"][meta.doc_id] = {"versions": {}, "active_version": None}
        self.data["docs"][meta.doc_id]["versions"][meta.version] = {
            "file_path": meta.file_path,
            "file_hash": meta.file_hash,
            "chunk_count": meta.chunk_count,
            "status": meta.status,
            "created_at": meta.created_at,
            "updated_at": meta.updated_at,
        }
        self.data["docs"][meta.doc_id]["active_version"] = meta.version
        self._save()

    def archive_old_versions(self, doc_id: str, keep_version: str):
        """将除 keep_version 外的所有版本标记为 archived"""
        doc = self.get_doc(doc_id)
        if not doc:
            return
        for ver, info in doc["versions"].items():
            if ver != keep_version:
                info["status"] = "archived"

    # ─── 分片 ───

    def get_chunks(self, doc_id: str, version: str) -> list[dict]:
        """获取指定文档版本的所有分片"""
        result = []
        for cid, meta in self.data["chunks"].items():
            if meta["doc_id"] == doc_id and meta["version"] == version:
                result.append({"chunk_id": cid, **meta})
        return result

    def register_chunks(self, chunks: list[ChunkMeta]):
        for c in chunks:
            self.data["chunks"][c.chunk_id] = {
                "doc_id": c.doc_id,
                "version": c.version,
                "content_hash": c.content_hash,
                "section": c.section,
                "chunk_index": c.chunk_index,
                "status": c.status,
            }
        self._save()

    def soft_delete_chunks(self, chunk_ids: list[str]):
        """软删除分片（标记 deleted，不物理删除）"""
        for cid in chunk_ids:
            if cid in self.data["chunks"]:
                self.data["chunks"][cid]["status"] = "deleted"
        self._save()

    def rollback_version(self, doc_id: str, target_version: str):
        """回滚到指定版本"""
        doc = self.get_doc(doc_id)
        if not doc or target_version not in doc["versions"]:
            raise ValueError(f"版本不存在: {doc_id}@{target_version}")
        self.data["docs"][doc_id]["active_version"] = target_version
        doc["versions"][target_version]["status"] = "active"
        self._save()
        logger.info(f"[IndexRegistry] 回滚: {doc_id} → {target_version[:8]}")

    def list_docs(self) -> list[dict]:
        """列出所有文档"""
        result = []
        for doc_id, doc in self.data["docs"].items():
            active_ver = doc.get("active_version")
            if active_ver and active_ver in doc["versions"]:
                info = doc["versions"][active_ver]
                result.append({
                    "doc_id": doc_id,
                    "active_version": active_ver[:8],
                    "file_path": info.get("file_path", ""),
                    "chunk_count": info.get("chunk_count", 0),
                    "updated_at": info.get("updated_at", ""),
                })
        return result


# ============================================================
# Hash 计算
# ============================================================

def _file_hash(file_path: str) -> str:
    """计算文件 MD5（4KB 分块读取，防大文件爆内存）"""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            md5.update(chunk)
    return md5.hexdigest()


def _content_hash(text: str) -> str:
    """计算文本内容 MD5（归一化后）"""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def _generate_doc_id(file_path: str) -> str:
    """根据文件路径生成稳定的 doc_id（同一文件始终相同）"""
    return hashlib.md5(Path(file_path).as_posix().encode('utf-8')).hexdigest()[:16]


def _generate_version(file_hash: str, timestamp: str = None) -> str:
    """生成版本号：hash[:12] + 时间戳"""
    ts = timestamp or str(int(time.time()))
    return f"{file_hash[:12]}_{ts}"


def _generate_chunk_id(doc_id: str, version: str, chunk_index: int, content: str, section: str = "") -> str:
    """生成稳定的 chunk_id"""
    base = f"{doc_id}_{version}_{section}_{chunk_index}"
    content_part = _content_hash(content)[:8]
    return f"{base}_{content_part}"[:100]  # Milvus VARCHAR 最长 100


# ============================================================
# diff 计算
# ============================================================

def compute_diff(
    old_chunks: list[dict],
    new_chunks: list[ChunkMeta],
) -> IndexDiff:
    """
    计算新旧版本的差异

    对比策略（按优先级）：
    1. 先按 section + chunk_index 匹配（同一位置的 chunk）
    2. 再按 content_hash 匹配（内容相同但位置变化的 chunk）
    3. 剩下的：旧的 = 删除，新的 = 新增
    """
    # 按位置匹配
    old_by_pos = {f"{c.get('section','')}_{c.get('chunk_index',0)}": c for c in old_chunks}
    new_by_pos = {f"{c.section}_{c.chunk_index}": c for c in new_chunks}

    diff = IndexDiff(
        doc_id=new_chunks[0].doc_id if new_chunks else "",
        old_version=old_chunks[0].get("version", "") if old_chunks else "",
        new_version=new_chunks[0].version if new_chunks else "",
        total_old=len(old_chunks),
        total_new=len(new_chunks),
    )

    matched_new_positions: set[str] = set()
    matched_old_positions: set[str] = set()

    # 第1轮：同位置 + 同 hash → 不变
    for pos, new_c in new_by_pos.items():
        if pos in old_by_pos:
            old_c = old_by_pos[pos]
            if old_c.get("content_hash") == new_c.content_hash:
                diff.unchanged.append(new_c.chunk_id)
                matched_new_positions.add(pos)
                matched_old_positions.add(pos)

    # 第2轮：同位置但 hash 不同 → 内容变化
    for pos, new_c in new_by_pos.items():
        if pos in matched_new_positions:
            continue
        if pos in old_by_pos:
            diff.changed.append(new_c.chunk_id)
            matched_new_positions.add(pos)
            matched_old_positions.add(pos)

    # 第3轮：剩余的新 chunk → 新增
    for pos, new_c in new_by_pos.items():
        if pos not in matched_new_positions:
            # 尝试按 content_hash 匹配
            old_hash_map = {c.get("content_hash"): c for c in old_chunks
                           if f"{c.get('section','')}_{c.get('chunk_index',0)}" not in matched_old_positions}
            if new_c.content_hash in old_hash_map:
                # 内容相同，位置变了 → 保留内容，视为不变
                diff.unchanged.append(new_c.chunk_id)
                matched_old_positions.add(
                    f"{old_hash_map[new_c.content_hash].get('section','')}_{old_hash_map[new_c.content_hash].get('chunk_index',0)}"
                )
            else:
                diff.added.append(new_c.chunk_id)

    # 第4轮：旧的 chunk 没被匹配 → 删除
    for pos, old_c in old_by_pos.items():
        if pos not in matched_old_positions:
            diff.deleted.append(old_c.get("chunk_id", ""))

    return diff


# ============================================================
# 增量索引管理器
# ============================================================

class IncrementalIndexer:
    """
    增量索引管理器

    使用：
        indexer = IncrementalIndexer()
        diff = indexer.update("manual_v2.pdf", new_chunks)
        # diff.summary → "abc12345→def67890: 不变=24, 新增=3, 删除=1, 变化=2"
    """

    def __init__(self, registry_path: str = "data/index_registry.json"):
        self.registry = IndexRegistry(registry_path)

    def update(
        self,
        file_path: str,
        new_chunks: list[Document],
        skip_if_unchanged: bool = True,
        doc_id: str = "",
        force_version: str = "",
        display_name: str = "",
    ) -> IndexDiff:
        """
        增量更新文档

        Args:
            file_path: 用于计算 MD5 的文件路径（实际存在的临时文件路径）
            new_chunks: 新版本的分片列表
            skip_if_unchanged: 文件 hash 没变则跳过
            doc_id: 指定 doc_id（为空则从 file_path 自动生成）
            force_version: 指定版本号（为空则自动生成）
            display_name: 显示用的文件名（为空则用 file_path）

        Returns:
            IndexDiff: 差异报告
        """
        doc_id = doc_id or _generate_doc_id(file_path)
        file_hash = _file_hash(file_path)
        new_version = force_version or _generate_version(file_hash)

        # 检查文件是否变化
        active_ver = self.registry.get_active_version(doc_id)
        if active_ver and skip_if_unchanged:
            doc_info = self.registry.get_doc(doc_id)
            if doc_info and active_ver in doc_info.get("versions", {}):
                old_file_hash = doc_info["versions"][active_ver].get("file_hash", "")
                if old_file_hash == file_hash:
                    logger.info(f"[IndexManager] 文件未变化，跳过: {Path(file_path).name}")
                    return IndexDiff(doc_id=doc_id, old_version=active_ver, new_version=active_ver)

        # 为新分片生成 ChunkMeta
        new_chunk_metas = []
        for i, doc in enumerate(new_chunks):
            section = doc.metadata.get("h1", "") or doc.metadata.get("h2", "") or ""
            cid = _generate_chunk_id(doc_id, new_version, i, doc.page_content, section)
            new_chunk_metas.append(ChunkMeta(
                chunk_id=cid,
                doc_id=doc_id,
                version=new_version,
                content_hash=_content_hash(doc.page_content),
                section=section,
                chunk_index=i,
            ))

        # 获取旧版本分片
        old_chunks = []
        if active_ver:
            old_chunks = self.registry.get_chunks(doc_id, active_ver)

        # 计算 diff
        diff = compute_diff(old_chunks, new_chunk_metas)

        # 执行更新
        # 1. 软删除旧分片
        if diff.deleted:
            self.registry.soft_delete_chunks(diff.deleted)

        # 2. 注册新分片
        self.registry.register_chunks(new_chunk_metas)

        # 3. 更新文档元数据
        doc_meta = DocMeta(
            doc_id=doc_id,
            version=new_version,
            file_path=display_name or file_path,
            file_hash=file_hash,
            chunk_count=len(new_chunks),
            status="active",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.registry.register_doc(doc_meta)

        # 4. 归档旧版本
        if active_ver:
            self.registry.archive_old_versions(doc_id, new_version)

        logger.info(f"[IndexManager] {Path(file_path).name}: {diff.summary}")

        # 返回实际要入库的分片（只返回 added + changed）
        to_index = []
        for c in new_chunk_metas:
            if c.chunk_id in diff.added or c.chunk_id in diff.changed:
                to_index.append(new_chunks[c.chunk_index])
        diff._to_index = to_index

        return diff

    def list_docs(self) -> list[dict]:
        return self.registry.list_docs()

    def get_diff(self, doc_id: str, new_version: str, old: list[ChunkMeta], new: list[ChunkMeta]) -> IndexDiff:
        return compute_diff(
            [{"chunk_id": c.chunk_id, "section": c.section, "chunk_index": c.chunk_index, "content_hash": c.content_hash, "version": c.version} for c in old],
            new,
        )


# ============================================================
# 全局实例
# ============================================================

incremental_indexer = IncrementalIndexer()


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import logging
    import tempfile
    logging.basicConfig(level=logging.INFO)

    # 创建临时测试文件
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="w") as tmp:
        tmp.write("test content")
        test_file = tmp.name

    # 模拟首次索引
    indexer = IncrementalIndexer(registry_path="data/test_registry.json")

    chunks_v1 = [
        Document(page_content="OSPF 协议配置步骤：1. 进入配置模式 2. 设置 router-id",
                 metadata={"h1": "OSPF配置"}),
        Document(page_content="BGP 邻居建立条件：AS号、TCP连通、路由可达",
                 metadata={"h1": "BGP配置"}),
    ]

    diff = indexer.update(test_file, chunks_v1)
    print(f"V1: {diff.summary}")

    # 模拟更新：创建新文件（内容不同）→ 保留第1条、修改第2条、新增第3条
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="w") as tmp2:
        tmp2.write("updated content")  # 不同内容 → 不同 MD5
        test_file2 = tmp2.name

    chunks_v2 = [
        Document(page_content="OSPF 协议配置步骤：1. 进入配置模式 2. 设置 router-id",
                 metadata={"h1": "OSPF配置"}),  # 不变
        Document(page_content="BGP 邻居建立条件：AS号匹配、TCP 179端口连通、路由可达、更新源可达",
                 metadata={"h1": "BGP配置"}),  # 内容变化（增加了细节）
        Document(page_content="VLAN 配置：access模式用于终端，trunk模式用于交换机互联",
                 metadata={"h1": "VLAN配置"}),  # 新增
    ]

    diff = indexer.update(test_file2, chunks_v2)
    print(f"V2: {diff.summary}")
    print(f"  不变: {len(diff.unchanged)}")
    print(f"  新增: {len(diff.added)}")
    print(f"  删除: {len(diff.deleted)}")
    print(f"  变化: {len(diff.changed)}")

    # 列出所有文档
    print(f"\n已索引文档:")
    for d in indexer.list_docs():
        print(f"  {d['doc_id']} v{d['active_version']} — {d['file_path']} ({d['chunk_count']} chunks)")

    # 清理测试文件
    os.remove("data/test_registry.json")
    os.unlink(test_file)
    os.unlink(test_file2)
    print("\n✅ 增量索引测试通过")
