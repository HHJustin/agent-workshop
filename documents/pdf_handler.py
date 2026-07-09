"""
PDF 处理模块 — PyPDFLoader + MinerU 双方案

设计思路：
    - 简单 PDF（纯文本、无表格/多栏）→ PyPDFLoader：速度优先，pip install 即用
    - 复杂 PDF（表格、公式、多栏排版、扫描件）→ MinerU：精度优先，保留结构

面试话术：
    "PDF 解析我做了分级处理。先用规则判断文档复杂度—
     简单文档走 PyPDFLoader 快速提取，复杂排版走 MinerU 做结构化解析。
     性能和精度兼顾，不是所有 PDF 都走慢路径。"

Author: 程响
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


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
        import magic_pdf.model as model_config
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.config.enums import SupportedPdfParseMethod

        # 读取 PDF
        dataset = PymuDocDataset(file_path)

        # 判断是否为扫描件（OCR 模式 vs 文本模式）
        if dataset.classify() == SupportedPdfParseMethod.OCR:
            logger.info(f"[MinerU] {Path(file_path).name}: OCR 模式")
            result = dataset.apply(dataset.ocr_model_choose())
        else:
            logger.info(f"[MinerU] {Path(file_path).name}: 文本解析模式")
            result = dataset.apply(dataset.txt_model_choose())

        # 提取 Markdown 内容
        md_content = result.get_content_in_md()

        # 包装为 LangChain Document
        document = Document(
            page_content=md_content if md_content else "",
            metadata={
                "_source": file_path,
                "_loader": "MinerU",
                "_file_name": Path(file_path).name,
                "_parse_mode": "ocr" if dataset.classify() == SupportedPdfParseMethod.OCR else "txt",
            },
        )

        logger.info(
            f"[MinerU] {Path(file_path).name}: 提取完成, 内容长度={len(md_content) if md_content else 0}"
        )

        return [document]

    except ImportError:
        # MinerU 未安装时，给出明确的安装指引
        raise ImportError(
            "MinerU 未安装。安装方式：\n"
            "  方式1: pip install magic-pdf\n"
            "  方式2: Docker — docker pull opendatalab/mineru\n"
            "  详见: https://github.com/opendatalab/MinerU\n"
            "如果不需要复杂 PDF 解析，可设置 fallback='pypdf' 跳过 MinerU"
        )
    except Exception as e:
        logger.error(f"[MinerU] 提取失败: {file_path}, 错误: {e}")
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
    PDF 处理器 — 自动选择 PyPDFLoader 或 MinerU

    使用示例：
        >>> handler = PDFHandler(fallback="pypdf")
        >>> docs = handler.process("document.pdf")

    fallback 参数：
        - "auto"（默认）：自动判断复杂度，选最优方案
        - "pypdf"：强制用 PyPDFLoader（MinerU 未安装时推荐）
        - "mineru"：强制用 MinerU（需要已安装 MinerU）
    """

    def __init__(self, fallback: str = "auto"):
        """
        Args:
            fallback: 降级策略
                - "auto": 自动判断
                - "pypdf": 强制 PyPDFLoader
                - "mineru": 强制 MinerU
        """
        if fallback not in ("auto", "pypdf", "mineru"):
            raise ValueError(f"不支持的 fallback 值: {fallback}，可选: auto / pypdf / mineru")
        self.fallback = fallback

    def process(self, file_path: str | Path) -> list[Document]:
        """
        处理 PDF 文件

        策略：
        - fallback="pypdf"：直接走 PyPDFLoader
        - fallback="mineru"：直接走 MinerU
        - fallback="auto"：先分析复杂度，简单走 PyPDFLoader，复杂走 MinerU；
          MinerU 不可用时自动降级到 PyPDFLoader
        """
        file_path = str(file_path)

        if self.fallback == "pypdf":
            return extract_text_pypdf(file_path)

        if self.fallback == "mineru":
            return extract_text_mineru(file_path)

        # fallback == "auto"
        complexity = get_pdf_complexity(file_path)

        if complexity.is_simple:
            logger.info(f"[PDFHandler] 简单文档，使用 PyPDFLoader: {Path(file_path).name}")
            return extract_text_pypdf(file_path)

        # 复杂文档 → 尝试 MinerU，失败则降级
        logger.info(
            f"[PDFHandler] 复杂文档({', '.join(complexity.reasons)})，"
            f"尝试 MinerU: {Path(file_path).name}"
        )
        try:
            return extract_text_mineru(file_path)
        except ImportError as e:
            logger.warning(
                f"[PDFHandler] MinerU 不可用({e})，降级到 PyPDFLoader。"
                f"表格/公式可能丢失，建议安装 MinerU: pip install magic-pdf"
            )
            return extract_text_pypdf(file_path)
        except Exception as e:
            logger.error(f"[PDFHandler] MinerU 失败({e})，降级到 PyPDFLoader")
            return extract_text_pypdf(file_path)


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
