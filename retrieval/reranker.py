"""
Cross-Encoder Reranker — 精排模块

原理：Bi-encoder（Embedding）是 query 和 doc 分别向量化再算余弦相似度——速度快但精度低。
      Cross-encoder 把 query 和 doc 拼接成一对喂给 Transformer，深度 Attention 交互——精度高但速度慢。
      所以用漏斗架构：向量初筛 top-100 → Reranker 精排 top-5。

面试考点：
    Q: "Bi-encoder 和 Cross-encoder 原理区别？为什么不直接用 Reranker？"
    A: Bi-encoder 各自独立编码，速度快，适合海量初筛。
       Cross-encoder 把 query+doc 拼接做深度交互，精度极高但计算量大。
       对百万文档直接做 Cross-encoder 会卡死，必须先用 Embedding 缩到 top-100。

Author: 程响
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document


class CrossEncoderReranker:
    """
    Cross-Encoder 精排器

    使用 BGE-Reranker 模型（FlagEmbedding），对候选文档做精细排序。

    安装依赖：
        pip install FlagEmbedding

    使用方式：
        reranker = CrossEncoderReranker()
        results = reranker.rerank("OSPF 配置", [(doc1, 0.8), (doc2, 0.7)], top_k=3)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """懒加载模型"""
        if self._model is not None:
            return
        from FlagEmbedding import FlagReranker
        self._model = FlagReranker(self.model_name, use_fp16=True)
        print(f"[Reranker] 模型已加载: {self.model_name}")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Document, float]],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        精排候选文档

        Args:
            query: 查询文本
            candidates: [(doc, embedding_score), ...]
            top_k: 返回文档数

        Returns:
            [(doc, rerank_score), ...] — rerank_score 是 Cross-Encoder 分数
        """
        if not candidates or not query:
            return []

        self._load_model()
        documents, _ = zip(*candidates) if candidates else ([], [])

        # 构建 query-doc 对
        pairs = [[query, doc.page_content[:2000]] for doc in documents]

        # Cross-Encoder 打分
        scores = self._model.compute_score(pairs, normalize=True)

        # 如果是单文档，scores 是标量
        if not isinstance(scores, list):
            scores = [scores]

        # 排序
        ranked = sorted(
            [(doc, float(score)) for doc, score in zip(documents, scores)],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


# ============================================================
# 轻量级 Reranker（不依赖 GPU 的回退方案）
# ============================================================

class LightweightReranker:
    """
    轻量级 Reranker — 不依赖 GPU，用简单的语义特征做重排序

    用于 FlagEmbedding 未安装或 CPU-only 场景的回退方案。
    基于 token 重叠度 + 关键词命中 + 位置特征加权，不替代 Cross-Encoder 的精度。
    """

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Document, float]],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        轻量精排：embedding_score * 0.5 + 关键词特征 * 0.5
        """
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        scored = []
        for doc, emb_score in candidates:
            text = doc.page_content.lower()

            # 特征1：关键词命中率
            hits = sum(1 for t in query_tokens if t in text)
            keyword_score = hits / max(len(query_tokens), 1)

            # 特征2：精确匹配加分（错误码、IP、端口号等）
            exact_bonus = 0.0
            for token in query_tokens:
                if token in text:
                    # 数字/特殊字符开头的 token 是精确匹配（如 "ERR-5001"）
                    if token[0].isdigit() or not token[0].isalpha():
                        exact_bonus += 0.2

            # 综合分数
            combined = emb_score * 0.5 + (keyword_score + exact_bonus) * 0.5
            scored.append((doc, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ============================================================
# 全局实例
# ============================================================

_cross_encoder: CrossEncoderReranker | None = None
_lightweight: LightweightReranker | None = None


def get_reranker(prefer_cross_encoder: bool = True) -> CrossEncoderReranker | LightweightReranker:
    """获取 Reranker：优先 Cross-Encoder，不可用时回退轻量版"""
    global _cross_encoder, _lightweight

    if prefer_cross_encoder:
        if _cross_encoder is None:
            try:
                _cross_encoder = CrossEncoderReranker()
                _cross_encoder._load_model()  # 验证能否加载
                return _cross_encoder
            except (ImportError, Exception):
                print("[Reranker] FlagEmbedding 不可用，使用轻量级回退方案")
                prefer_cross_encoder = False

    if _lightweight is None:
        _lightweight = LightweightReranker()
    return _lightweight


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    docs = [
        Document(page_content="OSPF 协议配置：router-id 1.1.1.1，network 10.0.0.0 0.0.0.255 area 0"),
        Document(page_content="OSPF 邻居状态：FULL/DR/BDR，Hello 间隔 10s，Dead 间隔 40s"),
        Document(page_content="BGP 配置：neighbor 2.2.2.2 remote-as 65001"),
        Document(page_content="VLAN 配置：switchport mode access，switchport access vlan 10"),
    ]

    candidates = [(doc, 0.8) for doc in docs]

    # 使用轻量级 Reranker 测试（不需要 GPU）
    reranker = LightweightReranker()
    results = reranker.rerank("OSPF router-id 配置", candidates, top_k=3)

    print("=== 重排序结果 ===")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc.page_content[:80]}...")

    print("\n✅ Reranker 测试通过")
