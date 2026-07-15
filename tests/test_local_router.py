"""本地路由器单元测试"""
from agents.local_router import local_route, keyword_match, tfidf_match


def test_keyword_diagnosis():
    intent, conf, src = local_route("核心交换机 CPU 飙到 95%")
    assert intent == "diagnosis"


def test_keyword_qa():
    intent, conf, src = local_route("怎么配置 OSPF 协议")
    assert intent == "qa"


def test_keyword_project_intro():
    intent, conf, src = local_route("介绍一下你的项目经历")
    assert intent == "qa"  # project_intro 强制映射到 qa


def test_tfidf_fallback():
    intent, conf, src = local_route("请告诉我一些网络知识")
    # 无关键词命中 → TF-IDF
    assert src == "tfidf"
