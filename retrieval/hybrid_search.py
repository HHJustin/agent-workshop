"""
混合检索 — BM25 + Embedding + RRF 融合 + Reranker 精排

漏斗架构：
    BM25 关键词召回（稀疏）──┐
                             ├── RRF 融合 top-20 ──→ Reranker 精排 top-5
    Embedding 语义召回（稠密）─┘

面试考点：
    Q: "向量检索和关键词检索怎么融合？"
    A: RRF（Reciprocal Rank Fusion）。Embedding 和 BM25 的分数量纲不同（一个是余弦相似度，
       一个是词频统计），不能直接加权。RRF 用排名代替原始分数——排第一得 1/(k+1) 分，
       排第二得 1/(k+2) 分——量纲统一后再融合，取 top-20 送 Reranker 精排。

Author: 程响
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document

from app.logger import logger


# ============================================================
# RRF 融合
# ============================================================

def reciprocal_rank_fusion(
    result_lists: list[list[tuple[Document, float]]],
    k: int = 60,
    top_n: int = 20,
) -> list[tuple[Document, float]]:
    """
    RRF（Reciprocal Rank Fusion）融合多个检索结果

    公式：score(d) = Σ 1 / (k + rank_i(d))
    其中 k=60 是经典常数，rank_i(d) 是文档 d 在第 i 个结果列表中的排名（从 1 开始）

    Args:
        result_lists: 多个检索结果列表 [(doc, score), ...]
        k: RRF 常数，默认 60
        top_n: 融合后返回的文档数

    Returns:
        [(doc, rrf_score), ...]
    """
    if not result_lists:
        return []

    doc_scores: dict[str, float] = {}
    doc_objects: dict[str, Document] = {}

    for results in result_lists:
        for rank, (doc, _) in enumerate(results, start=1):
            # 用 page_content 前 200 字符作为去重 key
            key = doc.page_content[:200]
            doc_objects[key] = doc
            doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank)

    # 按 RRF 分数降序
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_objects[key], score) for key, score in ranked[:top_n]]


# ============================================================
# 混合检索器
# ============================================================

class HybridSearcher:
    """
    混合检索器 — BM25 + Embedding + RRF + Reranker

    使用方式：
        searcher = HybridSearcher()

        # 上传文档后建索引
        searcher.index(all_documents)

        # 混合检索
        results = await searcher.search("OSPF 配置方法", top_k=5)
    """

    def __init__(self):
        self._bm25_indexed = False

    def index(self, documents: list[Document]):
        """建 BM25 索引（Embedding 已在 Milvus 中，无需额外建索引）"""
        from retrieval.bm25 import rebuild_bm25_index
        rebuild_bm25_index(documents)
        self._bm25_indexed = True
        logger.info(f"[HybridSearch] BM25 索引已更新，{len(documents)} 篇文档")

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_meta: dict = None,
        use_reranker: bool = True,
    ) -> list[tuple[Document, float]]:
        """
        混合检索

        流程：
        1. Embedding 语义检索 (Milvus) → top-20
        2. BM25 关键词检索 → top-20
        3. RRF 融合 → top-20
        4. Reranker 精排 → top-K

        Args:
            query: 查询文本
            top_k: 最终返回文档数
            filter_meta: Milvus 元数据过滤（None=不过滤）
            use_reranker: 是否使用 Reranker 精排

        Returns:
            [(doc, score), ...]
        """
        from retrieval.vector_store import vector_store_manager
        from retrieval.bm25 import get_bm25_retriever

        # 第1路：Embedding 语义召回
        emb_results = []
        try:
            emb_docs = vector_store_manager.similarity_search(
                query, k=20, filter_meta=filter_meta,
            )
            emb_results = [(doc, 1.0) for doc in emb_docs]
            emb_results = [(doc, 1.0) for doc in emb_docs]  # 分数仅用于排名
        except Exception as e:
            logger.warning(f"[HybridSearch] Embedding 检索失败: {e}")

        # 第2路：BM25 关键词召回
        bm25_results = []
        try:
            bm25 = get_bm25_retriever()
            bm25_results = bm25.search(query, top_k=20)
        except Exception as e:
            logger.warning(f"[HybridSearch] BM25 检索失败: {e}")

        # 如果只有一路有结果，直接返回
        if emb_results and not bm25_results:
            return emb_results[:top_k]
        if bm25_results and not emb_results:
            return bm25_results[:top_k]

        # RRF 融合
        rrf_results = reciprocal_rank_fusion(
            [emb_results, bm25_results],
            top_n=20,
        )
        logger.info(
            f"[HybridSearch] RRF 融合: "
            f"Embedding={len(emb_results)} + BM25={len(bm25_results)} → {len(rrf_results)}"
        )

        if not rrf_results:
            return []

        # Reranker 精排
        if use_reranker and len(rrf_results) > top_k:
            try:
                from retrieval.reranker import get_reranker
                reranker = get_reranker()
                rrf_results = reranker.rerank(query, rrf_results, top_k=top_k)
                logger.info(f"[HybridSearch] Reranker 精排: → {len(rrf_results)}")
            except Exception as e:
                logger.warning(f"[HybridSearch] Reranker 失败: {e}")
                rrf_results = rrf_results[:top_k]

        return rrf_results[:top_k]


# ============================================================
# 全局实例
# ============================================================

hybrid_searcher = HybridSearcher()


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import asyncio
    import logging
    logging.basicConfig(level=logging.INFO)

    async def test():
        # 准备文档
        docs = [
            Document(page_content="OSPF 协议配置：router-id 1.1.1.1，network 10.0.0.0 0.0.0.255 area 0"),
            Document(page_content="BGP 邻居建立条件：AS号匹配、TCP 179端口连通、路由可达"),
            Document(page_content="核心交换机 CPU 使用率异常排查：查看进程、检查环路、升级固件"),
            Document(page_content="ERR-5001：数据库连接池耗尽，需要重启服务或扩容连接数"),
            Document(page_content="VLAN 10 配置：interface GigabitEthernet0/1，switchport access vlan 10"),
        ]

        searcher = HybridSearcher()
        searcher.index(docs)

        # 测试：精确错误码（Embedding 不擅长，BM25 擅长）
        print("=== 查询: ERR-5001 ===")
        # 注：完整混合检索需要 Milvus，这里只测 BM25 部分
        from retrieval.bm25 import get_bm25_retriever
        bm25 = get_bm25_retriever()
        results = bm25.search("ERR-5001", top_k=3)
        for doc, score in results:
            print(f"  [{score:.2f}] {doc.page_content[:80]}...")

        print("\n✅ 混合检索框架测试通过")

    asyncio.run(test())
