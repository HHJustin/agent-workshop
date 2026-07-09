"""
统一文档加载器 — 自动识别文件类型并路由到对应处理器

支持的格式：TXT / MD / PDF
待扩展：PNG / JPG（图片 Caption 生成）

Author: 程响
"""

from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from .pdf_handler import PDFHandler, get_pdf_complexity, analyze_pdf_structure

# 支持的文件扩展名
ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def load_document(
    file_path: str | Path,
    pdf_fallback: str = "auto",
) -> list[Document]:
    """
    统一文档加载入口 — 自动识别文件类型

    Args:
        file_path: 文件路径
        pdf_fallback: PDF 处理策略
            - "auto"（默认）：自动判断复杂度，复杂走 MinerU，简单走 PyPDFLoader
            - "pypdf"：强制 PyPDFLoader
            - "mineru"：强制 MinerU

    Returns:
        list[Document]: LangChain Document 列表

    Raises:
        ValueError: 不支持的文件类型
        FileNotFoundError: 文件不存在
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"不支持的文件类型: '{ext}'。当前支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # TXT / Markdown
    if ext in (".txt", ".md", ".markdown"):
        return _load_text(file_path)

    # PDF
    if ext == ".pdf":
        handler = PDFHandler(fallback=pdf_fallback)
        return handler.process(str(file_path))

    # 理论上不会走到这里
    raise ValueError(f"未知错误: 无法处理文件 {file_path}")


def load_document_with_info(
    file_path: str | Path,
    pdf_fallback: str = "auto",
) -> tuple[list[Document], dict]:
    """
    加载文档 + 返回文件元信息

    Returns:
        (documents, info_dict)
        info_dict: {
            "file_name": str,
            "file_size_kb": float,
            "file_type": str,
            "document_count": int,
            "total_chars": int,
        }
    """
    documents = load_document(file_path, pdf_fallback)
    file_path = Path(file_path)

    info = {
        "file_name": file_path.name,
        "file_size_kb": round(file_path.stat().st_size / 1024, 1),
        "file_type": file_path.suffix.lower(),
        "document_count": len(documents),
        "total_chars": sum(len(doc.page_content) for doc in documents),
    }

    # PDF 额外信息
    if file_path.suffix.lower() == ".pdf":
        try:
            complexity = get_pdf_complexity(str(file_path))
            info["pdf_complexity"] = {
                "page_count": complexity.page_count,
                "has_table": complexity.has_table,
                "has_multi_column": complexity.has_multi_column,
                "is_scanned": complexity.is_scanned,
                "used_mineru": complexity.needs_mineru,
            }
        except Exception:
            pass  # 分析失败不阻塞加载

    return documents, info


def _load_text(file_path: Path) -> list[Document]:
    """加载 TXT / Markdown 文件"""
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()

    for doc in documents:
        doc.metadata["_source"] = str(file_path)
        doc.metadata["_loader"] = "TextLoader"
        doc.metadata["_file_name"] = file_path.name

    return documents


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("用法: python loader.py <file_path>")
        sys.exit(1)

    docs, info = load_document_with_info(sys.argv[1])
    print(f"\n文件信息: {info}")
    print(f"\n文档数量: {len(docs)}")
    for i, doc in enumerate(docs[:3]):
        print(f"\n--- Document {i+1} ---")
        print(doc.page_content[:300])
