"""
LLM and Embeddings factory.
Centralizes model construction; supports OpenAI, Anthropic, and Ollama (open-source).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Tuple

import structlog

from app.core.config import LLMProvider, get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


def build_llm_and_embeddings() -> Tuple[Any, Any]:
    """Return (llm, embeddings) for the configured provider."""
    provider = settings.LLM_PROVIDER

    if provider == LLMProvider.OPENAI:
        return _build_openai()
    elif provider == LLMProvider.ANTHROPIC:
        return _build_anthropic()
    elif provider == LLMProvider.OLLAMA:
        return _build_ollama()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _build_openai() -> Tuple[Any, Any]:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
            "Set it in .env or as an environment variable."
        )

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        api_key=settings.OPENAI_API_KEY,
        request_timeout=settings.LLM_REQUEST_TIMEOUT,
        streaming=True,
    )
    embeddings = OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    logger.info("llm_initialized", provider="openai", model=settings.OPENAI_MODEL)
    return llm, embeddings


def _build_anthropic() -> Tuple[Any, Any]:
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import OpenAIEmbeddings  # Anthropic doesn't provide embeddings

    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")

    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
    )
    # Fall back to OpenAI embeddings for Anthropic provider
    embeddings = OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    logger.info("llm_initialized", provider="anthropic")
    return llm, embeddings


def _build_ollama() -> Tuple[Any, Any]:
    """
    Ollama for fully local / open-source inference.
    Requires running: `ollama serve` and `ollama pull <model>`.
    Uses nomic-embed-text for local embeddings.
    """
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings

    llm = ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
    )
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=settings.OLLAMA_BASE_URL,
    )
    logger.info(
        "llm_initialized",
        provider="ollama",
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )
    return llm, embeddings
