from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# 项目路径配置：集中维护运行所需的默认文件和目录位置。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chat.sqlite3"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "schema_catalog.json"
DEFAULT_WEB_PATH = PROJECT_ROOT / "web" / "index.html"
DEFAULT_RAG_INDEX_PATH = PROJECT_ROOT / "data" / "rag_index"


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str
    api_key_env: str
    model: str
    base_url: str | None


@dataclass(frozen=True)
class LLMRuntimeConfig:
    provider: str
    api_key: str | None
    model: str
    base_url: str | None


# 主聊天模型配置：定义 Provider 白名单、默认模型和密钥/地址变量名。
LLM_PROVIDER_ENV = "LLM_PROVIDER"
DEFAULT_LLM_PROVIDER = "qwen"

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL = None

DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

QWEN_API_KEY_ENV = "QWEN_API_KEY"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

LLM_PROVIDER_CONFIGS = {
    "openai": LLMProviderConfig("openai", OPENAI_API_KEY_ENV, "gpt-4o-mini", OPENAI_BASE_URL),
    "deepseek": LLMProviderConfig("deepseek", DEEPSEEK_API_KEY_ENV, "deepseek-flash", DEEPSEEK_BASE_URL),
    "qwen": LLMProviderConfig("qwen", QWEN_API_KEY_ENV, "deepseek-v4.1-flash", QWEN_BASE_URL),
}


# RAG Embedding 配置：固定使用 Qwen 向量模型，密钥沿用 QWEN_API_KEY。
RAG_EMBEDDING_PROVIDER = "qwen"
RAG_EMBEDDING_MODEL = "text-embedding-v3"
RAG_EMBEDDING_LITELLM_MODEL = "dashscope/text-embedding-v3"
RAG_EMBEDDING_BASE_URL = QWEN_BASE_URL
RAG_EMBEDDING_API_KEY_ENV = QWEN_API_KEY_ENV


# 意图识别配置：定义分类器默认 Provider、Jev 密钥变量名和固定模型。
DEFAULT_INTENT_PROVIDER = "llm"
INTENT_PROVIDER_ENV = "INTENT_PROVIDER"

TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
JEV_MODEL = "typesafe/jev-1.13"


@dataclass(frozen=True)
class IntentRuntimeConfig:
    provider: str
    typesafe_api_key: str | None


@dataclass(frozen=True)
class RagEmbeddingRuntimeConfig:
    provider: str
    api_key: str | None
    model: str
    litellm_model: str
    base_url: str | None


@dataclass(frozen=True)
class NL2SQLRuntimeConfig:
    enabled: bool
    dialect: str
    database_url: str | None
    max_attempts: int
    max_reflections: int
    max_rows: int
    max_columns: int
    query_timeout_seconds: int
    max_sql_length: int


# 意图 LLM 配置：允许意图分类使用独立 Provider，未配置时跟随主聊天模型 Provider。
INTENT_LLM_PROVIDER_ENV = "INTENT_LLM_PROVIDER"
INTENT_LLM_MODEL_CONFIGS = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-flash",
    "qwen": "deepseek-v4.1-flash",
}


@dataclass(frozen=True)
class IntentLLMRuntimeConfig:
    provider: str
    api_key: str | None
    model: str
    base_url: str | None


# 运行参数配置：集中管理路由策略等不属于模型凭据的运行时参数。
INTENT_MIN_CONFIDENCE = 0.70


# NL2SQL 配置：业务库必须单独配置，绝不复用聊天会话 SQLite。
NL2SQL_ENABLED_ENV = "NL2SQL_ENABLED"
DEFAULT_NL2SQL_ENABLED = True
NL2SQL_DIALECT = "sqlite"
NL2SQL_DATABASE_URL_ENV = "NL2SQL_DATABASE_URL"
NL2SQL_MAX_SQL_ATTEMPTS = 3
NL2SQL_MAX_REFLECTIONS = 2
NL2SQL_MAX_ROWS = 100
NL2SQL_MAX_COLUMNS = 30
NL2SQL_QUERY_TIMEOUT_SECONDS = 5
NL2SQL_MAX_SQL_LENGTH = 12000


def get_llm_runtime_config() -> LLMRuntimeConfig:
    provider = (os.getenv(LLM_PROVIDER_ENV) or DEFAULT_LLM_PROVIDER).strip().lower()
    provider_config = LLM_PROVIDER_CONFIGS.get(provider)
    if provider_config is None:
        raise ValueError(provider)
    return LLMRuntimeConfig(
        provider=provider,
        api_key=os.getenv(provider_config.api_key_env),
        model=provider_config.model,
        base_url=provider_config.base_url,
    )


def get_rag_embedding_runtime_config() -> RagEmbeddingRuntimeConfig:
    return RagEmbeddingRuntimeConfig(
        provider=RAG_EMBEDDING_PROVIDER,
        api_key=os.getenv(RAG_EMBEDDING_API_KEY_ENV),
        model=RAG_EMBEDDING_MODEL,
        litellm_model=RAG_EMBEDDING_LITELLM_MODEL,
        base_url=RAG_EMBEDDING_BASE_URL,
    )


def get_intent_runtime_config() -> IntentRuntimeConfig:
    return IntentRuntimeConfig(
        provider=(os.getenv(INTENT_PROVIDER_ENV) or DEFAULT_INTENT_PROVIDER).strip().lower(),
        typesafe_api_key=os.getenv(TYPESAFE_API_KEY_ENV),
    )


def get_intent_llm_runtime_config() -> IntentLLMRuntimeConfig:
    provider = (
        os.getenv(INTENT_LLM_PROVIDER_ENV)
        or os.getenv(LLM_PROVIDER_ENV)
        or DEFAULT_LLM_PROVIDER
    ).strip().lower()
    provider_config = LLM_PROVIDER_CONFIGS.get(provider)
    if provider_config is None:
        raise ValueError(provider)
    return IntentLLMRuntimeConfig(
        provider=provider,
        api_key=os.getenv(provider_config.api_key_env),
        model=INTENT_LLM_MODEL_CONFIGS.get(provider, provider_config.model),
        base_url=provider_config.base_url,
    )


def get_nl2sql_runtime_config() -> NL2SQLRuntimeConfig:
    return NL2SQLRuntimeConfig(
        enabled=_read_bool(NL2SQL_ENABLED_ENV, DEFAULT_NL2SQL_ENABLED),
        dialect=NL2SQL_DIALECT,
        database_url=os.getenv(NL2SQL_DATABASE_URL_ENV),
        max_attempts=NL2SQL_MAX_SQL_ATTEMPTS,
        max_reflections=NL2SQL_MAX_REFLECTIONS,
        max_rows=NL2SQL_MAX_ROWS,
        max_columns=NL2SQL_MAX_COLUMNS,
        query_timeout_seconds=NL2SQL_QUERY_TIMEOUT_SECONDS,
        max_sql_length=NL2SQL_MAX_SQL_LENGTH,
    )


def _read_bool(env_name: str, default: bool) -> bool:
    value = os.getenv(env_name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{env_name} 必须是 true/false、1/0 或 yes/no。")
