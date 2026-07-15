"""
PDF 处理模块 — 四级处理管道

标准流程（遵循业界最佳实践）：
    1. 文本提取（Parsing）
       ├── 文本型 PDF → PyPDFLoader（快）或 pdfplumber（保留表格）
       ├── 表格密集型 → pdfplumber（保留行列结构）
       ├── 复杂排版/公式 → MinerU（结构化解析 + OCR）
       └── 扫描图片型 → OCR Pipeline（pdf2image → PaddleOCR/Tesseract）

    2. 清洗与结构化（Cleaning）→ cleaner.py 六步管道
       ├── 去噪：页眉页脚、页码、水印
       ├── 结构还原：基于字体大小识别标题层级
       ├── 表格转 Markdown/JSON
       └── 去重：目录页与正文重复识别

    3. 分块（Chunking）→ splitter.py
       ├── 按段落自然边界优先切
       ├── 超长段按字符切兜底
       └── 表格/代码块保护（不可分割原子块）

    4. 向量化与存储 → vector_store.py
       └── Embedding → Milvus

面试话术：
    "PDF 解析我做了分级处理。先用规则判断文档复杂度—
     纯文本走 PyPDFLoader（快），表格多走 pdfplumber（保留结构），
     复杂排版走 MinerU（精度优先），扫描件走 OCR。
     性能和精度兼顾，不是所有 PDF 都走慢路径。"

Author: 程响
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.logger import logger


# ============================================================
# 复杂度判定
# ============================================================

@dataclass
class PDFComplexity:
    """PDF 复杂度评估结果"""
    path: str
    page_count: int = 0
    has_table: bool = False
    has_multi_column: bool = False
    has_formula: bool = False
    is_scanned: bool = False
    needs_mineru: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def is_simple(self) -> bool:
        return not self.needs_mineru


def get_pdf_complexity(file_path: str | Path) -> PDFComplexity:
    """
    快速评估 PDF 复杂度，判断是否需要 MinerU

    检测方式：
    1. 用 PyPDFLoader 提取文本
    2. 分析文本特征（行宽差异大 → 多栏、表格分隔符 → 表格、数学符号 → 公式）

    Args:
        file_path: PDF 文件路径

    Returns:
        PDFComplexity: 复杂度评估结果
    """
    file_path = str(file_path)
    result = PDFComplexity(path=file_path)

    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
        pages = loader.load()
        result.page_count = len(pages)

        if result.page_count == 0:
            result.needs_mineru = True
            result.reasons.append("无法提取文本，可能是扫描件")
            return result

        # 取前 5 页做采样分析
        sample_pages = pages[:5]
        all_text = "\n".join(p.page_content for p in sample_pages)

        # 检测1：文本为空或过短 → 扫描件
        if len(all_text.strip()) < 50:
            result.is_scanned = True
            result.needs_mineru = True
            result.reasons.append("文本量极少，疑似扫描件")
            return result

        # 检测2：多栏布局（行宽差异 > 50%）
        line_widths = [len(line) for line in all_text.split("\n") if line.strip()]
        if line_widths and max(line_widths) > 1.8 * min(line_widths):
            result.has_multi_column = True

        # 检测3：表格特征（连续的 | 分隔符、对齐的空格列）
        pipe_lines = sum(1 for line in all_text.split("\n") if line.count("|") >= 2)
        if pipe_lines >= 3:
            result.has_table = True

        # 检测4：公式特征
        formula_chars = sum(1 for c in all_text if c in "∫∑∏√∂∞≈≠≤≥αβγθλμσ")
        if formula_chars >= 5:
            result.has_formula = True

        # 综合判断
        if result.has_table or result.has_multi_column or result.has_formula:
            result.needs_mineru = True
            parts = []
            if result.has_table:
                parts.append("含表格")
            if result.has_multi_column:
                parts.append("含多栏排版")
            if result.has_formula:
                parts.append("含数学公式")
            result.reasons.append("、".join(parts))

        logger.info(
            f"[PDF复杂度] {Path(file_path).name}: "
            f"页数={result.page_count}, 需MinerU={result.needs_mineru}, "
            f"原因={result.reasons or '简单文档'}"
        )

    except ImportError:
        logger.warning("PyPDFLoader 未安装，默认使用 MinerU")
        result.needs_mineru = True
        result.reasons.append("PyPDFLoader 不可用，降级到 MinerU")
    except Exception as e:
        logger.error(f"PDF 复杂度分析失败: {e}")
        result.needs_mineru = True
        result.reasons.append(f"分析失败({e})，保守选择 MinerU")

    return result


# ============================================================
# PyPDFLoader 方案（简单文档优先）
# ============================================================

def extract_text_pypdf(file_path: str | Path) -> list[Document]:
    """
    使用 PyPDFLoader 提取 PDF 文本

    适用场景：纯文本文档、报告、标准排版 PDF
    优势：pip install pypdf 即可，速度快，无额外依赖

    Args:
        file_path: PDF 文件路径

    Returns:
        list[Document]: LangChain Document 列表（每页一个 Document）
    """
    file_path = str(file_path)

    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # 补充元数据
        for doc in documents:
            doc.metadata["_source"] = file_path
            doc.metadata["_loader"] = "PyPDFLoader"
            doc.metadata["_file_name"] = Path(file_path).name

        logger.info(f"[PyPDFLoader] {Path(file_path).name}: {len(documents)} 页")
        return documents

    except ImportError:
        raise ImportError(
            "PyPDFLoader 需要安装: pip install pypdf langchain-community"
        )
    except Exception as e:
        logger.error(f"[PyPDFLoader] 提取失败: {file_path}, 错误: {e}")
        raise RuntimeError(f"PyPDFLoader 提取失败: {e}") from e


# ============================================================
# pdfplumber 方案（表格密集型文档）
# ============================================================

def extract_text_pdfplumber(file_path: str | Path) -> list[Document]:
    """
    使用 pdfplumber 提取 PDF 文本（保留表格行列结构）

    适用场景：产品手册、技术文档等含表格的 PDF
    优势：表格自动转 Markdown/JSON，保留行列结构；比 MinerU 轻量
    依赖：pip install pdfplumber

    Args:
        file_path: PDF 文件路径

    Returns:
        list[Document]: LangChain Document 列表（每页一个 Document）
    """
    file_path = str(file_path)

    try:
        import pdfplumber

        documents = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                parts = []

                # 1. 提取文本
                text = page.extract_text()
                if text:
                    parts.append(text)

                # 2. 提取表格并转为 Markdown
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    if table and len(table) > 1:
                        md_table = _table_to_markdown(table)
                        parts.append(f"\n【表格 {t_idx + 1}】\n{md_table}")

                content = "\n".join(parts)
                if content.strip():
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "_source": file_path,
                            "_loader": "pdfplumber",
                            "_file_name": Path(file_path).name,
                            "_page": i + 1,
                            "_tables": len(tables),
                        },
                    ))

        logger.info(f"[pdfplumber] {Path(file_path).name}: {len(documents)} 页")
        return documents

    except ImportError:
        raise ImportError("pdfplumber 需要安装: pip install pdfplumber")
    except Exception as e:
        logger.error(f"[pdfplumber] 提取失败: {file_path}, 错误: {e}")
        raise RuntimeError(f"pdfplumber 提取失败: {e}") from e


def _table_to_markdown(table: list[list]) -> str:
    """将二维表格列表转为 Markdown 格式"""
    if not table:
        return ""
    lines = []
    # 表头
    headers = [str(c) if c else "" for c in table[0]]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    # 数据行
    for row in table[1:]:
        cells = [str(c) if c else "" for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ============================================================
# OCR 方案（扫描图片型 PDF）
# ============================================================

def extract_text_ocr(file_path: str | Path, ocr_engine: str = "paddle") -> list[Document]:
    """
    使用 OCR 提取扫描图片型 PDF 的文本

    适用场景：扫描件、图片型 PDF（不可复制文本）
    流程：PDF → pdf2image 转图片 → OCR 识别 → 文本
    引擎：PaddleOCR（中文效果好）、Tesseract（通用）、EasyOCR

    Args:
        file_path: PDF 文件路径
        ocr_engine: OCR 引擎 paddle/tesseract/easyocr

    Returns:
        list[Document]: LangChain Document 列表
    """
    file_path = str(file_path)

    try:
        from pdf2image import convert_from_path

        images = convert_from_path(file_path, dpi=200)
        logger.info(f"[OCR] {Path(file_path).name}: {len(images)} 页图片待识别")

        ocr_func = _get_ocr_engine(ocr_engine)
        documents = []

        for i, img in enumerate(images):
            text = ocr_func(img)
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "_source": file_path,
                        "_loader": f"OCR/{ocr_engine}",
                        "_file_name": Path(file_path).name,
                        "_page": i + 1,
                    },
                ))

        logger.info(f"[OCR] {Path(file_path).name}: 识别完成, {len(documents)} 页有内容")
        return documents

    except ImportError as e:
        raise ImportError(
            f"OCR 依赖未安装: {e}。\n"
            "安装方式:\n"
            "  PaddleOCR: pip install paddlepaddle paddleocr pdf2image\n"
            "  Tesseract: pip install pytesseract pdf2image (需安装 tesseract-ocr)\n"
            "  EasyOCR:   pip install easyocr pdf2image"
        )
    except Exception as e:
        logger.error(f"[OCR] 识别失败: {file_path}, 错误: {e}")
        raise RuntimeError(f"OCR 识别失败: {e}") from e


def _get_ocr_engine(engine: str):
    """获取 OCR 引擎函数"""
    if engine == "paddle":
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(lang="ch")
            def _ocr(img):
                import numpy as np
                result = ocr.ocr(np.array(img), cls=False)
                if not result or not result[0]:
                    return ""
                return "\n".join(line[1][0] for line in result[0] if line)
            return _ocr
        except ImportError:
            raise ImportError("PaddleOCR 未安装")
    elif engine == "tesseract":
        import pytesseract
        return lambda img: pytesseract.image_to_string(img, lang="chi_sim+eng")
    elif engine == "easyocr":
        import easyocr
        reader = easyocr.Reader(["ch_sim", "en"])
        return lambda img: " ".join([item[1] for item in reader.readtext(img)])
    else:
        raise ValueError(f"不支持的 OCR 引擎: {engine}，可选: paddle/tesseract/easyocr")


# ============================================================
# MinerU 方案（复杂文档）
# ============================================================

def extract_text_mineru(file_path: str | Path) -> list[Document]:
    """
    使用 MinerU 提取 PDF 文本（保留表格/公式/多栏结构）

    适用场景：含表格、多栏排版、公式、扫描件的 PDF
    优势：结构保留好，支持 OCR
    依赖：MinerU 环境（GPU 推荐但非必须）

    MinerU 安装参考：
        pip install magic-pdf
        # 或使用 Docker 镜像：
        # docker run -v $(pwd):/data opendatalab/mineru:latest ...

    Args:
        file_path: PDF 文件路径

    Returns:
        list[Document]: LangChain Document 列表
    """
    file_path = str(file_path)

    try:
        import json, os, tempfile
        import magic_pdf.model as model_config
        from magic_pdf.tools.common import do_parse, prepare_env

        # 读取 PDF
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        # 设置输出目录
        output_dir = os.path.join(tempfile.gettempdir(), "mineru_output")
        filename = Path(file_path).stem

        local_image_dir, local_md_dir = prepare_env(output_dir, filename, "txt")

        model_config.__use_inside_model__ = True
        model_config.__model_mode__ = "lite"  # lite 模式跳过布局检测模型，避免依赖黑洞

        # 调用 do_parse
        do_parse(
            output_dir,
            filename,
            pdf_bytes,
            [],
            "txt",
            False,
            f_draw_span_bbox=False,
            f_draw_layout_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_json=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            f_draw_model_bbox=False,
        )

        # 读取生成的 Markdown
        md_path = os.path.join(local_md_dir, f"{filename}.md")
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
        else:
            md_content = ""

        document = Document(
            page_content=md_content if md_content else "",
            metadata={
                "_source": file_path,
                "_loader": "MinerU",
                "_file_name": Path(file_path).name,
            },
        )

        logger.info(
            f"[MinerU] {Path(file_path).name}: 提取完成, 内容长度={len(md_content) if md_content else 0}"
        )

        return [document]

    except Exception as e:
        logger.error(f"[MinerU] 提取失败: {file_path}, 错误: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"MinerU 提取失败: {e}") from e


# ============================================================
# PDF 结构分析（只分析元数据，不提取文本）
# ============================================================

def analyze_pdf_structure(file_path: str | Path) -> dict:
    """
    分析 PDF 结构信息（不提取全文，轻量级）

    Returns:
        dict: {
            "file_name": str,
            "page_count": int,
            "file_size_kb": float,
            "has_outline": bool,      # 是否有目录/书签
            "has_metadata": bool,      # 是否有文档元数据
            "estimated_tables": int,   # 预估表格数量
        }
    """
    file_path = str(file_path)
    result = {
        "file_name": Path(file_path).name,
        "file_size_kb": round(Path(file_path).stat().st_size / 1024, 1),
        "page_count": 0,
        "has_outline": False,
        "has_metadata": False,
        "estimated_tables": 0,
    }

    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
        pages = loader.load()
        result["page_count"] = len(pages)

        if pages:
            full_text = "\n".join(p.page_content for p in pages)
            result["estimated_tables"] = sum(
                1 for line in full_text.split("\n") if line.count("|") >= 2
            )

        logger.info(f"[PDF结构分析] {result}")
    except Exception as e:
        logger.warning(f"PDF 结构分析失败: {e}")
        result["error"] = str(e)

    return result


# ============================================================
# PDFHandler：带降级的统一入口
# ============================================================

class PDFHandler:
    """
    PDF 处理器 — 四级自动降级

    使用示例：
        >>> handler = PDFHandler(fallback="auto")
        >>> docs = handler.process("document.pdf")

    fallback 参数：
        - "auto"（默认）：自动判断 → 逐级降级
        - "pypdf"：强制 PyPDFLoader
        - "pdfplumber"：强制 pdfplumber（表格保留）
        - "mineru"：强制 MinerU
        - "ocr"：强制 OCR（扫描件专用）
    """

    def __init__(self, fallback: str = "auto"):
        if fallback not in ("auto", "pypdf", "pdfplumber", "mineru", "ocr"):
            raise ValueError(f"不支持的 fallback: {fallback}")
        self.fallback = fallback

    def process(self, file_path: str | Path) -> list[Document]:
        """
        处理 PDF 文件 — PyMuPDF (fitz)，多栏中文友好
        """
        file_path = str(file_path)

        import fitz
        docs = []
        doc = fitz.open(file_path)

        for i, page in enumerate(doc):
            # 按阅读顺序提取文本块（自动处理多栏）
            blocks = page.get_text("blocks")
            # 按 y 坐标排序，同高度按 x 排序
            blocks.sort(key=lambda b: (round(b[1] / 50) * 50, b[0]))

            text = "\n".join(b[4] for b in blocks if b[4].strip())
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "_source": file_path,
                        "_loader": "PyMuPDF",
                        "_file_name": Path(file_path).name,
                        "_page": i + 1,
                    },
                ))

        doc.close()
        chars = sum(len(d.page_content) for d in docs)
        logger.info(f"[PyMuPDF] {Path(file_path).name}: {len(docs)}页, {chars}字符")
        return docs


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("用法: python pdf_handler.py <pdf_path> [fallback=auto]")
        print("示例: python pdf_handler.py test.pdf")
        print("      python pdf_handler.py test.pdf pypdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    fallback = sys.argv[2] if len(sys.argv) > 2 else "auto"

    handler = PDFHandler(fallback=fallback)
    docs = handler.process(pdf_path)

    for i, doc in enumerate(docs):
        print(f"--- Document {i+1} (loader={doc.metadata.get('_loader')}) ---")
        print(doc.page_content[:500])
        print()
