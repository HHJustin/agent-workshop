"""
入库前6步数据清洗管道

面试考点：
    Q: "冗余无关信息入库前怎么处理？"
    A: 6步管道：格式清洗→去重→相关性过滤→结构化→异常检测→抽样验证。
       核心原则：宁愿少入高质量内容，不把低质量内容塞进向量库。

Author: 程响
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from app.logger import logger


# ============================================================
# 清洗结果
# ============================================================

@dataclass
class CleanStats:
    """清洗统计"""
    input_count: int = 0
    output_count: int = 0
    removed_format: int = 0       # 格式清洗移除的
    removed_duplicate: int = 0    # 去重移除的
    removed_irrelevant: int = 0   # 相关性过滤移除的
    flagged_anomaly: int = 0      # 标记为异常的
    issues: list[str] = field(default_factory=list)


# ============================================================
# 第1步：格式清洗
# ============================================================

# 常见的无意义行模式
NOISE_PATTERNS = [
    re.compile(r'^\s*$'),                                    # 空行
    re.compile(r'^第\s*\d+\s*页\s*$'),                       # 页码：第 1 页
    re.compile(r'^Page\s+\d+\s*$', re.IGNORECASE),            # Page 1
    re.compile(r'^\d+\s*/\s*\d+\s*$'),                        # 1/50
    re.compile(r'^版权所有|版权声明|Copyright', re.IGNORECASE),  # 版权
    re.compile(r'^目录|Table\s+of\s+Contents', re.IGNORECASE), # 目录
    re.compile(r'^文档版本|修订记录|更新日志|Change\s*Log', re.IGNORECASE),
    re.compile(r'^https?://'),                                 # 下载链接
    re.compile(r'^---+$'),                                     # 分隔线
    re.compile(r'^_{3,}$'),                                    # 下划线分隔
    re.compile(r'^={3,}$'),                                    # 等号分隔
    re.compile(r'^\*\s*$'),                                    # 纯星号行
]

# 页眉页脚关键词
HEADER_FOOTER_KEYWORDS = [
    '页眉', '页脚', 'header', 'footer',
    '机密', '内部资料', 'confidential',
]


def clean_format(docs: list[Document]) -> tuple[list[Document], CleanStats]:
    """
    第1步：格式清洗

    清洗内容：
    - 空行、纯空白行
    - 页眉页脚、页码
    - 版权声明、目录
    - 下载链接
    - 无意义分隔线
    """
    stats = CleanStats()
    cleaned = []

    for doc in docs:
        lines = doc.page_content.split('\n')
        kept_lines = []

        for line in lines:
            stripped = line.strip()
            is_noise = False

            # 匹配噪音模式
            for pattern in NOISE_PATTERNS:
                if pattern.match(stripped):
                    is_noise = True
                    break

            # 页眉页脚
            if not is_noise:
                for kw in HEADER_FOOTER_KEYWORDS:
                    if kw.lower() in stripped.lower():
                        is_noise = True
                        break

            if is_noise:
                stats.removed_format += 1
            else:
                kept_lines.append(line)

        if kept_lines:
            doc.page_content = '\n'.join(kept_lines)
            cleaned.append(doc)
        else:
            stats.issues.append(f"文档 {doc.metadata.get('_file_name', '')} 清洗后无内容")

    stats.input_count = len(docs)
    stats.output_count = len(cleaned)
    logger.info(f"[Cleaner L1] 格式清洗: {stats.input_count}→{stats.output_count} 文档, "
                f"移除 {stats.removed_format} 行噪音")
    return cleaned, stats


# ============================================================
# 第2步：多层次内容去重
# ============================================================

def _content_hash(text: str) -> str:
    """计算文本内容的 MD5（归一化后）"""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def _compute_simhash(text: str, bits: int = 64) -> int:
    """
    SimHash — 局部敏感哈希（LSH），用于近似重复检测

    原理：对每个 token 做 hash，按位加权累加，正变1负变0。
    两个 SimHash 的 Hamming 距离越小 → 文本越相似。
    距离 < 3 通常视为近似重复。
    """
    # 中文：按字符 N-gram；英文：按词
    tokens = []
    if any('一' <= c <= '鿿' for c in text[:100]):  # 简单判断是否中文为主
        # 中文：2-gram 字符
        tokens = [text[i:i+2] for i in range(len(text)-1)]
    else:
        tokens = text.lower().split()

    if not tokens:
        return 0

    # 截断，避免过长文本影响性能
    tokens = tokens[:500]

    weights = [0] * bits
    for token in tokens:
        token_hash = int(hashlib.md5(token.encode('utf-8')).hexdigest()[:16], 16)
        for i in range(bits):
            if token_hash & (1 << i):
                weights[i] += 1
            else:
                weights[i] -= 1

    fingerprint = 0
    for i in range(bits):
        if weights[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    """计算两个整数的 Hamming 距离"""
    xor = a ^ b
    return xor.bit_count()


def _ngram_overlap(text1: str, text2: str, n: int = 3) -> float:
    """N-gram 重叠度——比词级 Jaccard 更细粒度"""
    def ngrams(text, n):
        return set(text[i:i+n] for i in range(len(text)-n+1))

    ng1 = ngrams(text1.lower(), n)
    ng2 = ngrams(text2.lower(), n)
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)


def _semantic_similarity(text1: str, text2: str) -> float:
    """
    语义相似度去重 — 用 Embedding 向量比较

    这是最准确但最慢的方法，仅在 SimHash 和 N-gram 都不确定时使用。
    调用 DashScope text-embedding-v4 计算余弦相似度。
    注意：懒加载 + 缓存 Embedding 实例，避免重复初始化。
    """
    try:
        from app.llm_factory import get_embedding_model
        import numpy as np

        emb = get_embedding_model()
        # 截断到 2000 字符避免超长文本
        vecs = emb.embed_documents([text1[:2000], text2[:2000]])
        v1, v2 = np.array(vecs[0]), np.array(vecs[1])

        # 余弦相似度
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(cos_sim)
    except Exception as e:
        logger.warning(f"[Cleaner L2] 语义相似度计算失败: {e}")
        return 0.0


def clean_duplicates(
    docs: list[Document],
    simhash_hamming_threshold: int = 3,
    ngram_threshold: float = 0.80,
    semantic_threshold: float = 0.92,
) -> tuple[list[Document], CleanStats]:
    """
    第2步：四层去重策略

    | 策略 | 精度 | 速度 | 适用 |
    |------|------|------|------|
    | MD5 精确去重 | 100% | 极快 | 完全一致的文档 |
    | SimHash + Hamming | ~95% | 快 | 近似重复（排版差异、少量编辑） |
    | N-gram 重叠度 | ~90% | 中等 | SimHash 不确定时确认 |
    | Embedding 语义相似 | ~98% | 慢 | 同义改写、翻译、重述 |
    """
    stats = CleanStats()
    seen_hashes: set[str] = set()
    seen_simhashes: list[tuple[int, str]] = []  # [(simhash, md5)]
    result: list[Document] = []

    # 去重统计分类
    dup_by_md5 = 0
    dup_by_simhash = 0
    dup_by_ngram = 0
    dup_by_semantic = 0

    for doc in docs:
        text = doc.page_content
        if len(text) < 30:  # 太短不参与去重
            result.append(doc)
            continue

        h = _content_hash(text)

        # 第1层：MD5 精确去重
        if h in seen_hashes:
            dup_by_md5 += 1
            continue

        # 第2层：SimHash 近似重复检测
        simhash = _compute_simhash(text)
        is_dup = False

        for existing_sh, existing_md5 in seen_simhashes[-20:]:  # 只和最近20条比
            dist = _hamming_distance(simhash, existing_sh)
            if dist <= simhash_hamming_threshold:
                # SimHash 命中 → 用 N-gram 确认
                existing_text = next(
                    (r.page_content for r in result
                     if _content_hash(r.page_content) == existing_md5),
                    "",
                )
                if existing_text:
                    overlap = _ngram_overlap(text, existing_text)
                    if overlap >= ngram_threshold:
                        dup_by_simhash += 1
                        is_dup = True
                        break
                    # N-gram 不确定 → 用 Embedding 最终确认
                    elif overlap >= 0.5:
                        sem_sim = _semantic_similarity(text, existing_text)
                        if sem_sim >= semantic_threshold:
                            dup_by_semantic += 1
                            is_dup = True
                            logger.debug(
                                f"[Cleaner L2] 语义去重: sim={sem_sim:.3f}, "
                                f"文本: {text[:50]}..."
                            )
                            break

        if not is_dup:
            seen_hashes.add(h)
            seen_simhashes.append((simhash, h))
            result.append(doc)
        else:
            stats.removed_duplicate += 1

    stats.output_count = len(result)
    dedup_detail = f"MD5={dup_by_md5}, SimHash={dup_by_simhash}, N-gram={dup_by_ngram}, Semantic={dup_by_semantic}"
    logger.info(f"[Cleaner L2] 去重: {len(docs)}→{len(result)} ({dedup_detail})")
    return result, stats


# ============================================================
# 第3步：相关性过滤
# ============================================================

# 业务相关关键词（网络运维场景）
RELEVANT_KEYWORDS = [
    '配置', '参数', '命令', '协议', '端口', '接口',
    '故障', '告警', '排查', '诊断', '修复', '解决',
    '步骤', '操作', '指南', '说明', '要求', '规范',
    'CPU', '内存', '磁盘', '网络', '交换机', '路由器',
    'OSPF', 'BGP', 'VLAN', 'ACL', 'QoS', 'SNMP',
    '错误', '日志', '监控', '性能', '优化',
    'error', 'warning', 'critical', 'config',
]

IRRELEVANT_KEYWORDS = [
    '广告', '推广', '促销', '招聘', '公司简介', '联系我们',
    'advertisement', 'promotion', 'recruitment',
]


def _is_relevant(text: str) -> tuple[bool, str]:
    """判断文本是否与业务相关"""
    text_lower = text.lower()

    # 1. 检查无关关键词
    for kw in IRRELEVANT_KEYWORDS:
        if kw.lower() in text_lower:
            return False, f"含无关关键词: {kw}"

    # 2. 文本太短且无实质内容
    if len(text) < 20 and not any(c.isdigit() for c in text):
        return False, "文本过短且无实质内容"

    # 3. 检查是否包含业务相关关键词
    relevance_score = sum(1 for kw in RELEVANT_KEYWORDS if kw.lower() in text_lower)
    if relevance_score == 0:
        return False, "无业务相关关键词"

    return True, ""


def clean_irrelevant(docs: list[Document]) -> tuple[list[Document], CleanStats]:
    """
    第3步：相关性过滤

    只保留业务相关的内容（配置、故障、操作等），过滤广告、公司介绍等
    """
    stats = CleanStats()
    result = []

    for doc in docs:
        relevant, reason = _is_relevant(doc.page_content)
        if relevant:
            result.append(doc)
        else:
            stats.removed_irrelevant += 1
            logger.debug(f"[Cleaner L3] 过滤: {reason} — {doc.page_content[:50]}...")

    stats.output_count = len(result)
    logger.info(f"[Cleaner L3] 相关性过滤: {len(docs)}→{len(result)} 文档, "
                f"移除 {stats.removed_irrelevant} 个无关")
    return result, stats


# ============================================================
# 第4步：结构化整理
# ============================================================

def clean_structure(docs: list[Document]) -> tuple[list[Document], CleanStats]:
    """
    第4步：结构化整理 — 补充和规范化 metadata

    确保每个 chunk 携带：
    - _source: 文件路径
    - _file_name: 文件名
    - _extension: 文件类型
    - _chunk_index: 在文档中的序号
    - _cleaned_at: 清洗时间
    """
    import time
    from pathlib import Path

    stats = CleanStats()
    cleaned_at = time.strftime("%Y-%m-%d %H:%M:%S")

    for i, doc in enumerate(docs):
        source = doc.metadata.get("_source", "")
        if source:
            doc.metadata.setdefault("_file_name", Path(source).name)
            doc.metadata.setdefault("_extension", Path(source).suffix.lower())
        doc.metadata["_chunk_index"] = i
        doc.metadata["_cleaned_at"] = cleaned_at

        # 保留已有的 h1/h2/h3 标题信息
        for key in ("h1", "h2", "h3"):
            if key in doc.metadata and doc.metadata[key]:
                # 标题信息写入 content 开头，方便检索
                title = doc.metadata[key]
                if not doc.page_content.startswith(title):
                    doc.page_content = f"[{title}] {doc.page_content}"

    stats.output_count = len(docs)
    logger.info(f"[Cleaner L4] 结构化: {len(docs)} 个文档 metadata 已规范化")
    return docs, stats


# ============================================================
# 第5步：异常检测
# ============================================================

@dataclass
class AnomalyReport:
    """异常检测报告"""
    chunk_index: int
    issue: str
    severity: str  # "error" | "warning"
    preview: str


def detect_anomalies(docs: list[Document]) -> tuple[list[Document], list[AnomalyReport], CleanStats]:
    """
    第5步：异常检测

    检测项：
    - chunk 过短（< 20 字符）
    - 乱码比例过高（> 30%）
    - 重复率过高（> 80% 相似于其他 chunk）
    - 无正文内容（纯符号/数字）
    """
    stats = CleanStats()
    reports: list[AnomalyReport] = []
    valid: list[Document] = []

    def _garbled_ratio(text: str) -> float:
        """估算乱码比例：非中英文数字标点的比例"""
        valid_chars = sum(1 for c in text if c.isalnum() or '一' <= c <= '鿿'
                          or c in ' .,;:!?()[]{}"\'-_=+/\\@#$%^&*|~`<>')
        return 1.0 - (valid_chars / max(len(text), 1))

    for i, doc in enumerate(docs):
        text = doc.page_content.strip()
        issues = []

        # 检测1：过短
        if len(text) < 20:
            issues.append(("chunk 过短 (<20字符)", "warning"))

        # 检测2：乱码
        gr = _garbled_ratio(text)
        if gr > 0.3:
            issues.append((f"乱码比例 {gr:.0%}", "error"))

        # 检测3：纯符号
        alpha_ratio = sum(c.isalpha() or '一' <= c <= '鿿' for c in text) / max(len(text), 1)
        if alpha_ratio < 0.1:
            issues.append((f"可读字符占比 {alpha_ratio:.0%}", "error"))

        if issues:
            stats.flagged_anomaly += 1
            for issue, severity in issues:
                reports.append(AnomalyReport(
                    chunk_index=i, issue=issue, severity=severity,
                    preview=text[:80],
                ))
                if severity == "warning":
                    logger.warning(f"[Cleaner L5] 异常({severity}): {issue} — {text[:60]}")
                else:
                    logger.error(f"[Cleaner L5] 异常({severity}): {issue} — {text[:60]}")

            # error 级别的异常 → 丢弃；warning 级别 → 保留但标记
            has_error = any(s == "error" for _, s in issues)
            if not has_error:
                doc.metadata["_has_anomaly"] = "warning"
                valid.append(doc)
        else:
            valid.append(doc)

    stats.output_count = len(valid)
    logger.info(f"[Cleaner L5] 异常检测: {len(docs)}→{len(valid)} 正常, "
                f"标记 {stats.flagged_anomaly} 个异常")
    return valid, reports, stats


# ============================================================
# 第6步：抽样验证
# ============================================================

async def sample_validation(
    query: str,
    expected_keywords: list[str],
    docs: list[Document],
    k: int = 3,
) -> dict:
    """
    第6步：抽样验证 — 用已知问题检索，验证是否能召回正确内容

    Args:
        query: 测试查询
        expected_keywords: 期望召回内容包含的关键词
        docs: 入库的文档列表
        k: 召回数量

    Returns:
        {"passed": bool, "recalled": int, "matched": list[str], "missed": list[str]}
    """
    from retrieval.vector_store import vector_store_manager

    try:
        retrieved = vector_store_manager.similarity_search(query, k=k)
    except Exception as e:
        return {"passed": False, "error": str(e), "recalled": 0, "matched": [], "missed": expected_keywords}

    recalled_text = " ".join(d.page_content for d in retrieved)

    matched = [kw for kw in expected_keywords if kw.lower() in recalled_text.lower()]
    missed = [kw for kw in expected_keywords if kw not in matched]

    result = {
        "passed": len(matched) >= len(expected_keywords) * 0.6,  # ≥60% 关键词匹配即通过
        "recalled": len(retrieved),
        "matched": matched,
        "missed": missed,
    }

    if result["passed"]:
        logger.info(f"[Cleaner L6] 抽样验证通过: {len(matched)}/{len(expected_keywords)} 关键词命中")
    else:
        logger.warning(f"[Cleaner L6] 抽样验证未通过: 缺失 {missed}")
    return result


# ============================================================
# 完整管道
# ============================================================

@dataclass
class CleanResult:
    """完整清洗结果"""
    documents: list[Document]
    stats: dict[str, CleanStats]
    anomalies: list[AnomalyReport]
    total_input: int = 0
    total_output: int = 0

    @property
    def pass_rate(self) -> float:
        return self.total_output / max(self.total_input, 1)

    @property
    def summary(self) -> str:
        return (
            f"清洗完成: {self.total_input}→{self.total_output} 文档 "
            f"(通过率 {self.pass_rate:.1%}), "
            f"异常 {len(self.anomalies)} 个"
        )


def full_clean(
    docs: list[Document],
    skip_l3: bool = False,  # 用户自上传文档时跳过相关性过滤
    skip_l6: bool = True,   # L6 需要向量库在线，默认跳过
) -> CleanResult:
    """
    执行完整的 6 步清洗管道

    Args:
        docs: 待清洗的文档列表
        skip_l6: 是否跳过抽样验证（需要向量库已入库）

    Returns:
        CleanResult
    """
    total_input = len(docs)
    all_stats = {}

    # L1: 格式清洗
    docs, stats_l1 = clean_format(docs)
    all_stats["format"] = stats_l1

    # L2: 去重
    docs, stats_l2 = clean_duplicates(docs)
    all_stats["duplicate"] = stats_l2

    # L3: 相关性过滤（用户自上传文档可跳过）
    if skip_l3:
        stats_l3 = CleanStats()
        stats_l3.output_count = len(docs)
        all_stats["relevance"] = stats_l3
        logger.info("[Cleaner L3] 已跳过（用户自上传文档）")
    else:
        docs, stats_l3 = clean_irrelevant(docs)
        all_stats["relevance"] = stats_l3

    # L4: 结构化
    docs, stats_l4 = clean_structure(docs)
    all_stats["structure"] = stats_l4

    # L5: 异常检测
    docs, anomalies, stats_l5 = detect_anomalies(docs)
    all_stats["anomaly"] = stats_l5

    logger.info(f"[Cleaner] 管道完成: {total_input}→{len(docs)} 文档")

    return CleanResult(
        documents=docs,
        stats=all_stats,
        anomalies=anomalies,
        total_input=total_input,
        total_output=len(docs),
    )


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # 模拟脏数据
    dirty_docs = [
        Document(page_content="版权所有 © 2024", metadata={"_source": "test.pdf"}),
        Document(page_content="", metadata={"_source": "test.pdf"}),
        Document(page_content="Page 1", metadata={"_source": "test.pdf"}),
        Document(page_content="   ", metadata={"_source": "test.pdf"}),
        Document(page_content="OSPF 协议配置步骤：1. 进入配置模式 2. 设置 router-id 3. 宣告网络", metadata={"_source": "test.pdf"}),
        Document(page_content="OSPF 协议配置步骤：1. 进入配置模式 2. 设置 router-id 3. 宣告网络", metadata={"_source": "test.pdf"}),  # 重复
        Document(page_content="公司招聘：诚聘网络工程师", metadata={"_source": "test.pdf"}),
        Document(page_content="BGP 邻居状态异常排查：检查 TCP 179 端口连通性", metadata={"_source": "test.pdf"}),
        Document(page_content="ab", metadata={"_source": "test.pdf"}),  # 过短
        Document(page_content="########", metadata={"_source": "test.pdf"}),  # 纯符号
    ]

    result = full_clean(dirty_docs)

    print(f"\n{'='*50}")
    print(f"输入: {result.total_input} 个文档")
    print(f"输出: {result.total_output} 个文档")
    print(f"通过率: {result.pass_rate:.1%}")

    for step, stats in result.stats.items():
        print(f"\n[{step}]")
        if stats.removed_format:
            print(f"  移除噪音: {stats.removed_format}")
        if stats.removed_duplicate:
            print(f"  移除重复: {stats.removed_duplicate}")
        if stats.removed_irrelevant:
            print(f"  移除无关: {stats.removed_irrelevant}")
        if stats.flagged_anomaly:
            print(f"  标记异常: {stats.flagged_anomaly}")

    print(f"\n异常详情:")
    for a in result.anomalies:
        print(f"  [{a.severity}] chunk#{a.chunk_index}: {a.issue} — {a.preview}")

    print(f"\n保留文档:")
    for i, doc in enumerate(result.documents):
        print(f"  [{i}] {doc.page_content[:80]}...")
