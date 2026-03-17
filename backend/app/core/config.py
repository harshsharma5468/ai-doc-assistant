"""
Core configuration management using Pydantic Settings.
Supports environment-specific overrides via .env files.
"""
from __future__ import annotations

import secrets
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class VectorStoreType(str, Enum):
    FAISS = "faiss"
    CHROMA = "chroma"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"  # local open-source


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    APP_NAME: str = "AI Document Assistant"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = False
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ALLOWED_HOSTS: List[str] = ["*"]
    API_V1_PREFIX: str = "/api/v1"

    # ── Server ─────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    LOG_LEVEL: str = "INFO"

    # ── LLM ───────────────────────────────────────────────────────────────
    LLM_PROVIDER: LLMProvider = LLMProvider.OPENAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    ANTHROPIC_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_REQUEST_TIMEOUT: int = 60

    # ── Vector Store ───────────────────────────────────────────────────────
    VECTOR_STORE_TYPE: VectorStoreType = VectorStoreType.FAISS
    VECTOR_STORE_PATH: Path = Path("./data/vectorstore")
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    EMBEDDING_DIMENSION: int = 1536  # text-embedding-3-small
    SIMILARITY_TOP_K: int = 5
    SIMILARITY_SCORE_THRESHOLD: float = 0.3

    # ── Document Processing ────────────────────────────────────────────────
    UPLOAD_DIR: Path = Path("./data/uploads")
    MAX_UPLOAD_SIZE_MB: int = 50
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".md", ".html"]

    # ── Conversation Memory ────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CONVERSATION_TTL_SECONDS: int = 3600  # 1 hour
    MAX_CONVERSATION_HISTORY: int = 20
    MEMORY_WINDOW_SIZE: int = 5

    # ── RAG Pipeline ───────────────────────────────────────────────────────
    RETRIEVER_SEARCH_TYPE: str = "mmr"  # mmr | similarity | similarity_score_threshold
    RETRIEVER_FETCH_K: int = 20  # fetch before MMR re-ranking
    MMR_LAMBDA_MULT: float = 0.5  # diversity vs relevance balance
    ENABLE_RERANKING: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ENABLE_HYDE: bool = False  # Hypothetical Document Embeddings
    PROMPT_TEMPLATE_PATH: Path = Path("./app/core/prompts")

    # ── Evaluation ─────────────────────────────────────────────────────────
    EVAL_DATASET_PATH: Path = Path("./evaluation/datasets")
    RAGAS_LLM_MODEL: str = "gpt-4o-mini"
    ENABLE_TRACING: bool = True

    # ── Security ──────────────────────────────────────────────────────────
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    API_KEY_HEADER: str = "X-API-Key"

    # ── Observability ─────────────────────────────────────────────────────
    METRICS_ENABLED: bool = True
    METRICS_PATH: str = "/metrics"
    JAEGER_HOST: Optional[str] = None
    JAEGER_PORT: int = 6831
    SENTRY_DSN: Optional[str] = None

    @field_validator("VECTOR_STORE_PATH", "UPLOAD_DIR", "EVAL_DATASET_PATH", mode="before")
    @classmethod
    def create_dirs(cls, v: Any) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def openai_configured(self) -> bool:
        return bool(self.OPENAI_API_KEY)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
