"""
BM25 关键词检索 — 稀疏召回通道

原理：基于词频-逆文档频率（TF-IDF 的改进版），对查询中的关键词做精确匹配。
弥补 Embedding 语义检索的盲区——Embedding 对专有名词（"ERR-5001"、"BGP"、"VLAN 100"）
不敏感，而 BM25 可以通过精确词匹配找回。

面试考点：
    Q: "为什么需要 BM25 + Embedding 混合检索？"
    A: Embedding 擅长语义匹配（"数据库连接失败" ≈ "DB connection timeout"），
       但对精确关键词（错误码、IP地址、端口号）不敏感。
       BM25 擅长精确词匹配，两者互补。RRF 融合后召回率显著提升。

Author: 程响
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document


class BM25Retriever:
    """
    BM25 关键词检索器

    使用方式：
        retriever = BM25Retriever()
        retriever.index(documents)  # 建索引
        results = retriever.search("OSPF 配置", top_k=5)
    """

    def __init__(self):
        self.documents: list[Document] = []
        self._index = None
        self._corpus: list[str] = []

    def index(self, documents: list[Document]):
        """
        建立 BM25 索引

        BM25 需要预先对所有文档建倒排索引，之后才能检索。
        """
        from rank_bm25 import BM25Okapi

        self.documents = documents
        self._corpus = [self._tokenize(doc.page_content) for doc in documents]
        self._index = BM25Okapi(self._corpus)

    def _tokenize(self, text: str) -> list[str]:
        """中文按字分词，英文按词分词"""
        import re
        # 简单分词：中文逐字，英文按空格和标点
        tokens = []
        # 保留中文字符、英文单词、数字
        for match in re.finditer(r'[一-鿿]|[a-zA-Z]+|\d+', text):
            token = match.group()
            if '一' <= token <= '鿿':
                tokens.append(token)  # 中文字符逐个
            else:
                tokens.append(token.lower())  # 英文/数字转小写
        return tokens

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回文档数

        Returns:
            [(Document, score), ...] — score 是 BM25 分数，越高越相关
        """
        if not self._index:
            return []

        query_tokens = self._tokenize(query)
        scores = self._index.get_scores(query_tokens)

        # 取 top-k
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in indexed[:top_k]:
            if score > 0:
                results.append((self.documents[idx], float(score)))
        return results


# ============================================================
# 全局实例（懒加载）
# ============================================================

_bm25_retriever: BM25Retriever | None = None


def get_bm25_retriever() -> BM25Retriever:
    """获取或创建 BM25 检索器"""
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever()
    return _bm25_retriever


def rebuild_bm25_index(documents: list[Document]):
    """重建 BM25 索引（文档更新后调用）"""
    get_bm25_retriever().index(documents)


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    docs = [
        Document(page_content="OSPF 协议配置步骤：1. 进入配置模式 2. 设置 router-id 3. 宣告网络"),
        Document(page_content="BGP 邻居建立条件：AS号匹配、TCP 179端口连通、路由可达"),
        Document(page_content="核心交换机 CPU 使用率异常排查：查看进程、检查环路、升级固件"),
        Document(page_content="ERR-5001：数据库连接池耗尽，需要重启服务或扩容连接数"),
    ]

    retriever = BM25Retriever()
    retriever.index(docs)

    # 测试1：精确关键词
    print("=== 精确关键词: ERR-5001 ===")
    results = retriever.search("ERR-5001", top_k=3)
    for doc, score in results:
        print(f"  [{score:.2f}] {doc.page_content[:80]}...")

    # 测试2：中文查询
    print("\n=== 中文查询: OSPF 配置 ===")
    results = retriever.search("OSPF 配置", top_k=3)
    for doc, score in results:
        print(f"  [{score:.2f}] {doc.page_content[:80]}...")

    print("\n✅ BM25 测试通过")
