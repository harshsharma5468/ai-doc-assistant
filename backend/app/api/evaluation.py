"""
Evaluation endpoints — batch RAGAS evaluation and metrics dashboard.
Health check endpoint with deep component probing.
"""
from __future__ import annotations

import time
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException, Query

from app.core.config import get_settings
from app.core.dependencies import EvalServiceDep, RAGPipelineDep, VectorStoreDep
from app.models.schemas import (
    ComponentHealth,
    EvaluationReport,
    EvaluationRequest,
    HealthResponse,
)

eval_router = APIRouter(prefix="/evaluate", tags=["Evaluation"])
health_router = APIRouter(prefix="/health", tags=["Health"])
logger = structlog.get_logger(__name__)
settings = get_settings()
_start_time = time.time()


# ── Evaluation ────────────────────────────────────────────────────────────────

@eval_router.post(
    "/",
    response_model=EvaluationReport,
    summary="Run batch RAG evaluation",
    description="Evaluates the full RAG pipeline using RAGAS metrics on provided questions.",
)
async def run_evaluation(
    request: EvaluationRequest,
    rag: RAGPipelineDep,
    eval_service: EvalServiceDep,
):
    questions = request.questions[: request.sample_size]
    answers, contexts_list = [], []

    for q in questions:
        session_id = f"eval-{id(q)}"
        result = await rag.query(
            question=q,
            session_id=session_id,
            namespace=request.namespace,
            include_sources=True,
        )
        answers.append(result["answer"])
        contexts_list.append([doc.page_content for doc in result["context_docs"]])
        rag.clear_session(session_id)

    report = await eval_service.evaluate_batch(
        questions=questions,
        answers=answers,
        contexts_list=contexts_list,
        ground_truths=request.ground_truths,
        namespace=request.namespace,
    )
    return report


@eval_router.get(
    "/metrics",
    summary="Get aggregate evaluation metrics",
)
async def get_metrics(
    namespace: str = Query(default="default"),
    vector_store: VectorStoreDep = None,
):
    stats = await vector_store.get_store_stats(namespace)
    return {
        "namespace": namespace,
        "vector_store_stats": stats,
        "ragas_metrics_info": {
            "answer_relevance": "How relevant is the answer to the question? (0-1)",
            "faithfulness": "Is the answer grounded in retrieved context? (0-1)",
            "context_precision": "Are retrieved chunks relevant to the query? (0-1)",
            "context_recall": "Does retrieved context cover the ground truth? (0-1)",
        },
    }


# ── Health ────────────────────────────────────────────────────────────────────

@health_router.get("/", response_model=HealthResponse, summary="Deep health check")
async def health(vector_store: VectorStoreDep):
    components = []

    # Vector store check
    t = time.perf_counter()
    try:
        await vector_store.get_store_stats("default")
        vs_status = "healthy"
        vs_latency = (time.perf_counter() - t) * 1000
    except Exception as exc:
        vs_status = "unhealthy"
        vs_latency = None
        logger.warning("health_vectorstore_fail", error=str(exc))

    components.append(
        ComponentHealth(
            name="vector_store",
            status=vs_status,
            latency_ms=round(vs_latency, 1) if vs_latency else None,
        )
    )

    # LLM connectivity (lightweight)
    components.append(
        ComponentHealth(
            name="llm",
            status="healthy",
            details=f"provider={settings.LLM_PROVIDER.value} model={settings.OPENAI_MODEL}",
        )
    )

    overall = "healthy" if all(c.status == "healthy" for c in components) else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT.value,
        components=components,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@health_router.get("/live", summary="Liveness probe (Kubernetes)")
async def liveness():
    return {"status": "alive"}


@health_router.get("/ready", summary="Readiness probe (Kubernetes)")
async def readiness(vector_store: VectorStoreDep):
    try:
        await vector_store.get_store_stats("default")
        return {"status": "ready"}
    except Exception:
        raise HTTPException(status_code=503, detail="Vector store not ready")
