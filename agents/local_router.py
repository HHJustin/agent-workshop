"""
本地意图路由器 — 纯本地判断，不调 LLM API

两级：
    L1: 关键词规则（0 延迟，覆盖 60%）  ← 和 IntentRouter 相同
    L2: TF-IDF + 余弦相似度（轻量，无需 GPU） ← 新增，替代 LLM 调用

优势：
    - 零 API 成本：不用调 qwen-turbo 做意图判断
    - < 1ms 延迟：纯本地计算
    - 隐私安全：用户 prompt 不离开服务器

待升级（需 pip install lightgbm scikit-learn）：
    - LightGBM 分类器替代 TF-IDF

Author: 程响
"""

import re
import math
from collections import Counter


# ==================== L1: 关键词规则（和 IntentRouter 共享） ====================

FAST_ROUTES = {
    "project_intro": [
        "项目经历", "简历", "个人介绍", "自我介绍", "智能运维",
        "故障诊断系统", "Agent Workshop", "OnCall",
    ],
    "diagnosis": [
        "告警", "故障", "异常", "报错", "超时", "宕机", "挂了", "不通",
        "CPU", "内存", "磁盘", "网络中断", "排查", "诊断",
    ],
    "report": [
        "生成报告", "周报", "月报", "汇总", "总结", "导出", "报表",
    ],
    "qa": [
        "怎么", "什么是", "如何", "为什么", "介绍一下", "说明", "解释",
        "配置", "参数", "协议", "步骤", "教程",
    ],
}


def keyword_match(query: str) -> tuple[str, float] | None:
    """关键词匹配 → (intent, confidence)"""
    q = query.lower()
    for intent, keywords in FAST_ROUTES.items():
        for kw in keywords:
            if kw.lower() in q:
                return intent, 0.90
    return None


# ==================== L2: TF-IDF 相似度（本地轻量） ====================

# 每个意图的"典型查询"语料
INTENT_CORPUS = {
    "project_intro": [
        "介绍一下你的项目经历",
        "简历上写了什么",
        "你叫什么名字",
        "你的技能有哪些",
    ],
    "diagnosis": [
        "核心交换机 CPU 飙到 95%",
        "服务器连不上了报 502",
        "帮我排查网络故障",
        "端口 flapping 告警",
    ],
    "report": [
        "生成这个月的网络运行报告",
        "汇总最近一周的告警",
        "导出系统运行状态报表",
        "做一份本月故障统计",
    ],
    "qa": [
        "OSPF 协议怎么配置",
        "什么是 BGP",
        "HTTP 状态码 502 是什么意思",
        "怎么排查 CPU 使用率高",
    ],
}

_idf_cache = None


def _compute_tf(doc: str) -> Counter:
    """词频统计（中英文混合分词）"""
    # 中文按2-gram字符，英文按空格分词
    tokens = []
    # 英文词
    tokens.extend(re.findall(r'[a-zA-Z]+', doc.lower()))
    # 中文2-gram
    chinese = re.findall(r'[一-鿿]+', doc)
    for chunk in chinese:
        tokens.extend(chunk[i:i+2] for i in range(len(chunk)-1))
    return Counter(tokens)


def _compute_idf() -> dict[str, float]:
    """计算 IDF（全局只需算一次）"""
    global _idf_cache
    if _idf_cache:
        return _idf_cache

    N = 0
    df = Counter()
    all_docs = []
    for docs in INTENT_CORPUS.values():
        all_docs.extend(docs)

    N = len(all_docs)
    for doc in all_docs:
        unique_terms = set(_compute_tf(doc).keys())
        df.update(unique_terms)

    _idf_cache = {
        term: math.log((N + 1) / (freq + 1)) + 1
        for term, freq in df.items()
    }
    return _idf_cache


def _tfidf_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    """计算文本的 TF-IDF 向量"""
    tf = _compute_tf(text)
    total = sum(tf.values()) or 1
    return {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}


def _cosine_similarity(v1: dict[str, float], v2: dict[str, float]) -> float:
    """余弦相似度"""
    terms = set(v1) | set(v2)
    dot = sum(v1.get(t, 0) * v2.get(t, 0) for t in terms)
    norm1 = math.sqrt(sum(v ** 2 for v in v1.values())) or 1
    norm2 = math.sqrt(sum(v ** 2 for v in v2.values())) or 1
    return dot / (norm1 * norm2)


def tfidf_match(query: str) -> tuple[str, float]:
    """TF-IDF 相似度匹配 → (intent, confidence)"""
    idf = _compute_idf()
    qv = _tfidf_vector(query, idf)

    best_intent, best_score = "qa", 0.0
    for intent, docs in INTENT_CORPUS.items():
        # 取该意图所有文档的最高相似度
        max_sim = max(_cosine_similarity(qv, _tfidf_vector(d, idf)) for d in docs)
        if max_sim > best_score:
            best_score = max_sim
            best_intent = intent

    return best_intent, min(best_score, 0.95)


# ==================== 统一入口 ====================

def local_route(query: str) -> tuple[str, float, str]:
    """
    本地路由（两级）

    Returns:
        (intent, confidence, matched_by)
    """
    # L1: 关键词
    result = keyword_match(query)
    if result:
        intent, conf = result
        if intent == "project_intro":
            return "qa", conf, "keyword"
        return intent, conf, "keyword"

    # L2: TF-IDF
    intent, conf = tfidf_match(query)
    return intent, conf, "tfidf"
