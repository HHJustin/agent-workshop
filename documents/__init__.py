"""文档处理模块 — Agent Workshop

支持的文件类型：
    - TXT / Markdown：标准文本解析
    - PDF：PyPDFLoader（快速）+ MinerU（复杂排版）
    - PNG / JPG：Caption 生成（待实现，当前路径为展示选型能力预留）

Author: 程响
"""

from .loader import load_document
from .pdf_handler import (PDFHandler, get_pdf_complexity, extract_text_pypdf,
                          extract_text_mineru, analyze_pdf_structure)

__all__ = [
    "load_document",
    "PDFHandler",
    "get_pdf_complexity",
    "extract_text_pypdf",
    "extract_text_mineru",
    "analyze_pdf_structure",
]
