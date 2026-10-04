from __future__ import annotations

from typing import Any, Protocol

import litellm

from app.config import RagEmbeddingRuntimeConfig, get_rag_embedding_runtime_config


class RagServiceUnavailable(RuntimeError):
    """Raised when the configured embedding service cannot serve a request."""


class RagConfigurationError(RagServiceUnavailable):
    """Raised when required RAG configuration is missing."""


class EmbeddingAdapter(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector for each input text."""


class QwenEmbeddingAdapter:
    def __init__(self, config: RagEmbeddingRuntimeConfig) -> None:
        self.config = config

    @classmethod
    def from_env(cls) -> "QwenEmbeddingAdapter":
        return cls(get_rag_embedding_runtime_config())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.config.api_key:
            raise RagConfigurationError("缺少 RAG Embedding 配置：QWEN_API_KEY。")
        try:
            response = litellm.embedding(
                model=self.config.litellm_model,
                input=texts,
                api_key=self.config.api_key,
                api_base=self.config.base_url,
            )
            data = response["data"] if isinstance(response, dict) else getattr(response, "data", [])
            vectors = [list(item["embedding"] if isinstance(item, dict) else item.embedding) for item in data]
        except RagServiceUnavailable:
            raise
        except Exception as exc:
            raise RagServiceUnavailable("RAG Embedding 服务不可用。") from exc
        if len(vectors) != len(texts) or not all(vector for vector in vectors):
            raise RagServiceUnavailable("RAG Embedding 返回了无效向量。")
        return [[float(value) for value in vector] for vector in vectors]


class FakeEmbedding:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self.vector) for _ in texts]
