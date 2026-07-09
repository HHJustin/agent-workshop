"""
向量库抽象层 — Chroma（轻量）/ Milvus（生产）可切换

使用方式：
    from retrieval.vector_store import VectorStoreManager
    vs = VectorStoreManager()
    vs.add_documents(docs)
    retriever = vs.get_retriever(k=3)

面试考点：
    Q: "Chroma 和 Milvus 什么区别？为什么选 Milvus？"
    A: Chroma 内嵌式（SQLite），适合原型和小规模。Milvus 独立服务，分布式架构，
       支持百亿级向量、多种索引（IVF_FLAT/HNSW/DiskANN）、健康检查和监控。
       生产环境（7×24 运维系统）必须走 Milvus。

Author: 程响
"""

from __future__ import annotations

import uuid
import time
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import config
from app.llm_factory import get_embedding_model
from app.logger import logger


class VectorStoreManager:
    """
    向量库管理器 — 懒加载，首次调用时连接 Milvus
    """

    COLLECTION_NAME: str = "agent_workshop"

    def __init__(self):
        self.embedding: Embeddings = get_embedding_model()
        self._vector_store = None
        self._collection = None
        self._initialized = False
        logger.info(f"[VectorStore] 管理器就绪（懒加载模式），后端={config.vector_store}")

    def _ensure_initialized(self):
        """懒初始化 — 首次调用时才连接 Milvus/Chroma"""
        if self._initialized:
            return
        backend = config.vector_store
        if backend == "milvus":
            self._init_milvus()
        elif backend == "chroma":
            self._init_chroma()
        else:
            raise ValueError(f"不支持的向量库: {backend}")
        self._initialized = True
        logger.info(f"[VectorStore] 初始化完成: backend={backend}, collection={self.COLLECTION_NAME}")

    @staticmethod
    def _patch_milvus_alias():
        """修复 langchain_milvus 与 pymilvus ORM 的连接别名不一致问题"""
        if getattr(VectorStoreManager._patch_milvus_alias, "_done", False):
            return
        try:
            from pymilvus.milvus_client.milvus_client import MilvusClient
            _orig_init = MilvusClient.__init__

            def _wrapped_init(self, *args, **kwargs):
                _orig_init(self, *args, **kwargs)
                self._using = "default"

            MilvusClient.__init__ = _wrapped_init
            setattr(VectorStoreManager._patch_milvus_alias, "_done", True)
        except ImportError:
            pass

    # ==================== Milvus ====================

    def _init_milvus(self):
        """初始化 Milvus（复用 OnCall 项目的方案：IVF_FLAT, L2 距离, 1024维）"""
        # 兼容补丁：langchain_milvus 内部用自定义别名，ORM 需要 default
        self._patch_milvus_alias()

        from pymilvus import (
            Collection, CollectionSchema, DataType, FieldSchema,
            connections, utility, MilvusException,
        )

        host = config.milvus_host
        port = str(config.milvus_port)

        # 建立连接
        try:
            connections.connect(alias="default", host=host, port=port)
            logger.info(f"[Milvus] 已连接 {host}:{port}")
        except Exception as e:
            raise RuntimeError(f"Milvus 连接失败 ({host}:{port}): {e}")

        # 检查 / 创建 Collection
        if not utility.has_collection(self.COLLECTION_NAME):
            self._create_milvus_collection()
        else:
            self._check_milvus_dimension()

        self._collection = Collection(self.COLLECTION_NAME)
        self._load_milvus_collection()

        # 包装为 LangChain Milvus
        from langchain_milvus import Milvus
        self._vector_store = Milvus(
            embedding_function=self.embedding,
            collection_name=self.COLLECTION_NAME,
            connection_args={"host": host, "port": config.milvus_port},
            auto_id=False,
            text_field="content",
            vector_field="vector",
            primary_field="id",
            metadata_field="metadata",
        )

    def _create_milvus_collection(self):
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=config.embedding_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8000),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields=fields, description="Agent Workshop knowledge base")
        Collection(name=self.COLLECTION_NAME, schema=schema)

        # IVF_FLAT 索引
        from pymilvus import Collection
        col = Collection(self.COLLECTION_NAME)
        col.create_index(
            field_name="vector",
            index_params={
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )
        logger.info(f"[Milvus] 创建 collection: {self.COLLECTION_NAME}, embedding_dim={config.embedding_dim}")

    def _check_milvus_dimension(self):
        """检查已有 collection 的维度是否匹配，不匹配则重建"""
        from pymilvus import Collection, utility
        try:
            col = Collection(self.COLLECTION_NAME)
            for field in col.schema.fields:
                if field.name == "vector" and hasattr(field, "params"):
                    existing_dim = field.params.get("dim", 0)
                    if existing_dim and existing_dim != config.embedding_dim:
                        logger.warning(
                            f"[Milvus] 维度不匹配！已有={existing_dim}, 当前={config.embedding_dim}。重建中..."
                        )
                        utility.drop_collection(self.COLLECTION_NAME)
                        self._create_milvus_collection()
                        break
        except Exception:
            pass

    def _load_milvus_collection(self):
        from pymilvus import Collection, MilvusException, utility
        col = Collection(self.COLLECTION_NAME)
        try:
            load_state = utility.load_state(self.COLLECTION_NAME)
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                col.load()
                logger.info(f"[Milvus] collection '{self.COLLECTION_NAME}' 已加载到内存")
        except AttributeError:
            try:
                col.load()
            except MilvusException as e:
                if "already loaded" not in str(e).lower():
                    raise

    # ==================== Chroma（备用） ====================

    def _init_chroma(self):
        import os
        from langchain_chroma import Chroma
        os.makedirs(config.chroma_persist_dir, exist_ok=True)
        self._vector_store = Chroma(
            collection_name=config.chroma_collection_name,
            embedding_function=self.embedding,
            persist_directory=config.chroma_persist_dir,
        )

    # ==================== 公共接口 ====================

    def add_documents(self, documents: List[Document]) -> List[str]:
        """批量添加文档到向量库"""
        self._ensure_initialized()
        if not documents:
            return []
        start = time.time()
        ids = [str(uuid.uuid4()) for _ in documents]
        result = self._vector_store.add_documents(documents, ids=ids)
        elapsed = time.time() - start
        logger.info(f"[VectorStore] 入库 {len(documents)} 个文档, 耗时 {elapsed:.1f}s")
        return result

    def delete_by_source(self, file_path: str) -> int:
        """按文件来源删除（覆盖更新的前置步骤）"""
        self._ensure_initialized()
        if config.vector_store == "milvus":
            try:
                expr = f'metadata["_source"] == "{file_path}"'
                result = self._collection.delete(expr)
                deleted = result.delete_count if hasattr(result, "delete_count") else 0
                logger.info(f"[Milvus] 删除 {deleted} 条旧记录: {file_path}")
                return deleted
            except Exception as e:
                logger.warning(f"[Milvus] 删除失败（可能是首次索引）: {e}")
                return 0
        else:
            # Chroma 不支持按 metadata 删除，跳过
            return 0

    def get_retriever(self, k: int = None) -> VectorStoreRetriever:
        """获取向量检索器"""
        self._ensure_initialized()
        k = k or config.retrieval_top_k
        return self._vector_store.as_retriever(search_kwargs={"k": k})

    def similarity_search(
        self, query: str, k: int = None,
        filter_meta: dict = None,
    ) -> List[Document]:
        """相似文档检索（绕过 langchain_milvus，用 pymilvus 原生 API）"""
        self._ensure_initialized()
        k = k or config.retrieval_top_k

        if config.vector_store == "milvus":
            return self._search_native(query, k, filter_meta or {})
        elif config.vector_store == "chroma":
            if filter_meta:
                return self._search_with_filter_chroma(query, k, filter_meta)
            return self._vector_store.similarity_search(query, k=k)
        else:
            return self._vector_store.similarity_search(query, k=k)

    def _search_native(
        self, query: str, k: int, filter_meta: dict
    ) -> List[Document]:
        """用 pymilvus 原生 API 检索（绕过 langchain_milvus 兼容问题）"""
        from pymilvus import Collection

        query_vec = self.embedding.embed_query(query)
        col = Collection(self.COLLECTION_NAME)
        col.load()

        # 构建过滤表达式
        conditions = []
        for key, val in filter_meta.items():
            conditions.append(f'metadata["{key}"] == "{val}"')
        expr = " && ".join(conditions) if conditions else None

        if expr:
            logger.info(f"[VectorStore] 原生检索: {expr}")

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = col.search(
            data=[query_vec],
            anns_field="vector",
            param=search_params,
            limit=k * 3 if expr else k,
            expr=expr,
            output_fields=["id", "content", "metadata"],
        )

        docs = []
        for hits in results:
            for hit in hits:
                meta = hit.entity.get("metadata", {}) or {}
                content = hit.entity.get("content", "")
                docs.append(Document(page_content=content, metadata=meta))
                if len(docs) >= k:
                    break

        logger.info(f"[VectorStore] 检索完成: {len(docs)} 条")
        return docs

    def _search_with_filter_milvus(
        self, query: str, k: int, filter_meta: dict
    ) -> List[Document]:
        """Milvus 标量过滤 + 向量检索"""
        if not filter_meta:
            return self._vector_store.similarity_search(query, k=k)

        # 先试 expr 参数（新版 langchain_milvus）
        conditions = []
        for key, val in filter_meta.items():
            conditions.append(f'metadata["{key}"] == "{val}"')
        expr = " && ".join(conditions)

        logger.info(f"[VectorStore] Milvus 过滤检索: {expr}")

        try:
            docs = self._vector_store.similarity_search(query, k=k, expr=expr)
            if docs:
                return docs
            logger.info(f"[VectorStore] expr 返回0条，走手动过滤")
        except Exception as e:
            logger.warning(f"[VectorStore] expr 检索失败: {e}")

        # 回退：直接调 Milvus 底层 API 检索（绕过 langchain_milvus 兼容问题）
        try:
            from pymilvus import Collection
            query_vec = self.embedding.embed_query(query)
            col = Collection(self.COLLECTION_NAME)
            col.load()
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = col.search(
                data=[query_vec], anns_field="vector",
                param=search_params, limit=k * 5,
                output_fields=["id", "content", "metadata"],
            )
            docs = []
            for hits in results:
                for hit in hits:
                    meta = hit.entity.get("metadata", {}) or {}
                    content = hit.entity.get("content", "")
                    # 手动过滤
                    if filter_meta:
                        match = all(str(meta.get(mk, "")) == str(mv) for mk, mv in filter_meta.items())
                        if not match:
                            continue
                    docs.append(Document(page_content=content, metadata=meta))
                    if len(docs) >= k:
                        break
            if docs:
                logger.info(f"[VectorStore] 底层直搜: {len(docs)} 条")
            return docs
        except Exception as e2:
            logger.warning(f"[VectorStore] 底层直搜也失败: {e2}")

        return []

    def _search_with_filter_chroma(
        self, query: str, k: int, filter_meta: dict
    ) -> List[Document]:
        """Chroma 元数据过滤检索"""
        return self._vector_store.similarity_search(
            query, k=k, filter=filter_meta,
        )

    def get_retriever(
        self, k: int = None, filter_meta: dict = None
    ) -> VectorStoreRetriever:
        """获取向量检索器（支持元数据过滤）"""
        self._ensure_initialized()
        k = k or config.retrieval_top_k
        search_kwargs = {"k": k}
        if filter_meta:
            search_kwargs["filter"] = filter_meta
        return self._vector_store.as_retriever(search_kwargs=search_kwargs)

    @property
    def backend(self) -> str:
        return config.vector_store

    @property
    def is_healthy(self) -> bool:
        """健康检查"""
        try:
            # 简单测试：搜一个不可能命中的词
            self._vector_store.similarity_search("__health_check__", k=1)
            return True
        except Exception:
            return False


# 全局单例
vector_store_manager = VectorStoreManager()
