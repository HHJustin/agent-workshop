"""
统一配置管理 — Pydantic Settings + .env

使用方式：
    from app.config import config
    print(config.llm_provider)  # "dashscope" | "deepseek" | "vllm"

.env 文件示例（放在 agent_workshop/ 根目录）：
    LLM_PROVIDER=dashscope
    DASHSCOPE_API_KEY=sk-xxx
    DEEPSEEK_API_KEY=sk-xxx
    DEEPSEEK_BASE_URL=https://api.deepseek.com
    VLLM_BASE_URL=http://localhost:8000/v1

Author: 程响
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==================== 应用配置 ====================
    app_name: str = "Agent Workshop"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 9900

    # ==================== LLM 配置 ====================
    # 可选: "dashscope" | "deepseek" | "vllm"
    llm_provider: Literal["dashscope", "deepseek", "vllm"] = "dashscope"
    llm_model: str = "qwen-max"

    # DashScope（阿里云千问）
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"

    # 本地 vLLM（或其他 OpenAI 兼容服务）
    vllm_api_key: str = "not-needed"
    vllm_base_url: str = "http://localhost:8000/v1"

    # ==================== Embedding 配置 ====================
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024
    embedding_provider: Literal["dashscope", "vllm"] = "dashscope"

    # ==================== 向量库配置 ====================
    vector_store: Literal["chroma", "milvus"] = "milvus"
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "agent_workshop"
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # ==================== RAG 配置 ====================
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_size: int = 300
    retrieval_top_k: int = 3

    # ==================== Agent 配置 ====================
    agent_max_steps: int = 8         # Plan-Execute-Replan 最大步数
    agent_max_retries: int = 3       # 工具调用最大重试次数
    agent_retry_base_delay: float = 1.0  # 重试基础延迟（秒）
    agent_tool_execution: Literal["parallel", "sequential"] = "parallel"

    # ==================== 会话配置 ====================
    session_db_path: str = "data/chat_history.db"   # AsyncSqliteSaver

    # ==================== 日志配置 ====================
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_retention: str = "7 days"

    # ==================== 工具配置 ====================
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "test"
    prometheus_url: str = "http://localhost:9090"
    tavily_api_key: str = ""
    feishu_webhook_url: str = ""
    dingtalk_webhook_url: str = ""

    # ==================== Langfuse 配置 ====================
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_base_url: str = ""  # 兼容 LANGFUSE_BASE_URL 环境变量

    # ==================== MCP 配置 ====================
    mcp_enabled: bool = True
    mcp_network_url: str = "http://127.0.0.1:8005/mcp"

    @property
    def mcp_servers(self) -> dict:
        if not self.mcp_enabled:
            return {}
        return {
            "network": {
                "transport": "streamable-http",
                "url": self.mcp_network_url,
            },
        }

    # ==================== 跨域配置 ====================
    cors_origins: list[str] = ["*"]


# 全局单例
config = Settings()

# ChatQwen 需要环境变量
import os
os.environ.setdefault("DASHSCOPE_API_KEY", config.dashscope_api_key)

if __name__ == "__main__":
    print(f"Provider: {config.llm_provider}")
    print(f"Model: {config.llm_model}")
    print(f"Vector Store: {config.vector_store}")
    print(f"Chunk Size: {config.chunk_size}")
