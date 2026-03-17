"""
FastAPI dependency injection.
All services are constructed once at startup and injected via Depends().
"""
from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import Depends

from app.services.document_processor import DocumentProcessor
from app.services.evaluation import EvaluationService
from app.services.llm_factory import build_llm_and_embeddings
from app.services.rag_pipeline import RAGPipeline
from app.services.vector_store import VectorStoreService

logger = structlog.get_logger(__name__)

# Module-level singletons (initialized at app startup)
_llm = None
_embeddings = None
_vector_store: VectorStoreService | None = None
_rag_pipeline: RAGPipeline | None = None
_doc_processor: DocumentProcessor | None = None
_eval_service: EvaluationService | None = None


def initialize_services() -> None:
    """Called from app lifespan. Builds all services once."""
    global _llm, _embeddings, _vector_store, _rag_pipeline, _doc_processor, _eval_service

    logger.info("initializing_services")
    _llm, _embeddings = build_llm_and_embeddings()
    _vector_store = VectorStoreService(embeddings=_embeddings)
    _rag_pipeline = RAGPipeline(llm=_llm, vector_store=_vector_store)
    _doc_processor = DocumentProcessor(vector_store=_vector_store)
    _eval_service = EvaluationService(llm=_llm, embeddings=_embeddings)
    logger.info("services_ready")


# ── FastAPI Dependency Functions ───────────────────────────────────────────────

def get_vector_store() -> VectorStoreService:
    assert _vector_store is not None, "Services not initialized"
    return _vector_store


def get_rag_pipeline() -> RAGPipeline:
    assert _rag_pipeline is not None, "Services not initialized"
    return _rag_pipeline


def get_doc_processor() -> DocumentProcessor:
    assert _doc_processor is not None, "Services not initialized"
    return _doc_processor


def get_eval_service() -> EvaluationService:
    assert _eval_service is not None, "Services not initialized"
    return _eval_service


# ── Annotated Dependency Aliases ───────────────────────────────────────────────
VectorStoreDep = Annotated[VectorStoreService, Depends(get_vector_store)]
RAGPipelineDep = Annotated[RAGPipeline, Depends(get_rag_pipeline)]
DocProcessorDep = Annotated[DocumentProcessor, Depends(get_doc_processor)]
EvalServiceDep = Annotated[EvaluationService, Depends(get_eval_service)]
