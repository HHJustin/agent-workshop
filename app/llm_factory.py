"""
LLM 工厂 — 统一管理 DashScope / DeepSeek / vLLM 三种模型后端

使用方式：
    from app.llm_factory import get_chat_model, get_embedding_model
    llm = get_chat_model()           # 根据 config.llm_provider 自动选
    llm = get_chat_model("deepseek") # 手动指定

面试考点：
    Q: "怎么支持多模型切换？"
    A: 工厂模式 + Pydantic 配置驱动。config.llm_provider 决定用哪个后端，
       调用方不感知差异。切换模型只需改 .env 一行。

Author: 程响
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import config
from app.logger import logger


# ============================================================
# ChatModel 工厂
# ============================================================

def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    streaming: bool = True,
) -> BaseChatModel:
    """
    获取 LLM 实例

    支持的后端：
    - dashscope: 阿里云千问（OpenAI 兼容接口）
    - deepseek:  DeepSeek（OpenAI 兼容接口）
    - vllm:      本地 vLLM 服务

    Args:
        provider: 后端名称，默认取 config.llm_provider
        model: 模型名，默认取 config.llm_model
        temperature: 温度
        streaming: 是否流式
    """
    provider = provider or config.llm_provider
    model = model or config.llm_model

    if provider == "dashscope":
        _check_api_key(config.dashscope_api_key, "DASHSCOPE_API_KEY")
        logger.info(f"[LLM] DashScope/{model} (OpenAI 兼容模式)")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            api_key=config.dashscope_api_key,
            base_url=config.dashscope_base_url,
        )

    elif provider == "deepseek":
        _check_api_key(config.deepseek_api_key, "DEEPSEEK_API_KEY")
        logger.info(f"[LLM] DeepSeek/{model} (base_url={config.deepseek_base_url})")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            api_key=config.deepseek_api_key,        # type: ignore[arg-type]
            base_url=config.deepseek_base_url,
        )

    elif provider == "vllm":
        logger.info(f"[LLM] vLLM/{model} (base_url={config.vllm_base_url})")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            api_key=config.vllm_api_key,
            base_url=config.vllm_base_url,
        )

    else:
        raise ValueError(f"不支持的 LLM provider: {provider}，可选: dashscope / deepseek / vllm")


# ============================================================
# Embedding 工厂
# ============================================================

def get_embedding_model() -> Embeddings:
    """
    获取 Embedding 模型

    DashScope 使用自定义封装（兼容 dimensions 参数）
    本地 vLLM 使用 OpenAI 兼容接口
    """
    provider = config.embedding_provider

    if provider == "dashscope":
        logger.info(f"[Embedding] DashScope/{config.embedding_model} ({config.embedding_dim}维)")
        return DashScopeEmbeddings(
            api_key=config.dashscope_api_key,
            model=config.embedding_model,
            dimensions=config.embedding_dim,
        )

    elif provider == "vllm":
        logger.info(f"[Embedding] vLLM/{config.embedding_model}")
        return OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.vllm_api_key,
            base_url=config.vllm_base_url,
        )

    else:
        raise ValueError(f"不支持的 Embedding provider: {provider}")


# ============================================================
# DashScope Embedding 自定义封装
# ============================================================

class DashScopeEmbeddings(Embeddings):
    """
    DashScope text-embedding-v4 封装 — 处理 OpenAIEmbeddings 不兼容的问题

    面试考点：
        Q: "为什么不用 OpenAIEmbeddings？"
        A: DashScope 的 dimensions 参数和请求格式与 OpenAI 不完全兼容，
           langchain_openai 的 OpenAIEmbeddings 直接调会报 InvalidParameter。
           所以实现了 LangChain Embeddings 标准接口，调 openai SDK 的底层 API。
    """

    def __init__(self, api_key: str, model: str = "text-embedding-v4", dimensions: int = 1024):
        from openai import OpenAI

        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY 未设置")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model
        self.dimensions = dimensions
        logger.info(
            f"[Embedding] DashScopeEmbeddings 就绪: model={model}, dim={dimensions}, "
            f"key={api_key[:8]}...{api_key[-4:]}"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文档"""
        if not texts:
            return []
        # 过滤空字符串
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return []
        logger.info(f"[Embedding] 批量嵌入 {len(texts)} 个文档")
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """嵌入单个查询"""
        if not text or not text.strip():
            raise ValueError("查询文本不能为空")
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return response.data[0].embedding


# ============================================================
# 辅助
# ============================================================

def _check_api_key(key: str, name: str):
    """启动时检查 API Key，缺失给出明确提示但不阻止启动（方便本地 vLLM 场景）"""
    if not key or key == "":
        logger.warning(
            f"[LLM] {name} 未设置！"
            f"请在 .env 文件中配置 {name}=your_api_key"
        )


def get_llm_info() -> dict:
    """返回当前 LLM 配置摘要（前端/健康检查展示）"""
    return {
        "provider": config.llm_provider,
        "model": config.llm_model,
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
    }
