"""
文档分割模块 — 沿用 OnCall 项目的三阶段语义分块方案

三阶段：
    1. MarkdownHeaderTextSplitter：按 # / ## 标题边界切
    2. RecursiveCharacterTextSplitter：超长块二次字符分割
    3. _merge_small_chunks：合并 < 300 字符的碎片

Author: 程响
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# 配置（可后续移到 config.py）
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 300
SECONDARY_CHUNK_SIZE = 1600  # 二次分割用更大的 chunk

# Markdown 标题分割器
MARKDOWN_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
    ],
    strip_headers=False,
)

# 递归字符分割器
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=SECONDARY_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", "。", ".", "！", "？", " ", ""],
)


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> List[Document]:
    """
    统一的文档分割入口

    Args:
        documents: LangChain Document 列表
        chunk_size: 分块大小
        chunk_overlap: 重叠大小
        min_chunk_size: 最小分块大小（小于此值的碎片会被合并）

    Returns:
        分割后的 Document 列表
    """
    final_docs = []

    for doc in documents:
        source = doc.metadata.get("_source", "")
        ext = Path(source).suffix.lower() if source else ""

        if ext in (".md", ".markdown"):
            # Markdown → 语义分块
            chunks = _split_markdown(doc)
        else:
            # 普通文本（TXT / PDF 提取的内容等）→ 字符分割
            chunks = _split_text(doc)

        # 合并小碎片
        chunks = _merge_small_chunks(chunks, min_size=min_chunk_size)

        final_docs.extend(chunks)

    return final_docs


def _split_markdown(doc: Document) -> List[Document]:
    """Markdown 三阶段分割"""
    content = doc.page_content
    if not content or not content.strip():
        return []

    # 阶段1：按标题分割
    md_docs = MARKDOWN_SPLITTER.split_text(content)

    # 阶段2：超长块二次分割
    docs = TEXT_SPLITTER.split_documents(md_docs)

    # 保留原始元数据
    for d in docs:
        d.metadata.update(doc.metadata)

    return docs


def _split_text(doc: Document) -> List[Document]:
    """普通文本分割"""
    content = doc.page_content
    if not content or not content.strip():
        return []

    docs = TEXT_SPLITTER.create_documents(
        texts=[content],
        metadatas=[doc.metadata],
    )
    return docs


def _merge_small_chunks(
    documents: List[Document],
    min_size: int = MIN_CHUNK_SIZE,
    max_chunk_size: int = CHUNK_SIZE * 2,
) -> List[Document]:
    """
    合并太小的分片，防止碎片化

    策略：如果当前 chunk < min_size 且与下一个合并后不超过 max_chunk_size，则合并
    """
    if not documents:
        return []

    merged = []
    current = documents[0]

    for doc in documents[1:]:
        if (
            len(current.page_content) < min_size
            and len(current.page_content) + len(doc.page_content) < max_chunk_size
        ):
            current.page_content += "\n\n" + doc.page_content
        else:
            merged.append(current)
            current = doc

    merged.append(current)
    return merged
