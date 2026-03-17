"""
FastAPI application factory.
Configures middleware, mounts routers, registers lifespan events.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.api.chat import router as chat_router
from app.api.documents import router as docs_router
from app.api.evaluation import eval_router, health_router
from app.core.config import get_settings
from app.core.dependencies import initialize_services
from app.core.logging import configure_logging

configure_logging()
logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Prometheus Metrics ────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
RAG_QUERY_LATENCY = Histogram(
    "rag_query_duration_seconds",
    "RAG query latency",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 15.0, 30.0],
)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("app_startup", version=settings.APP_VERSION, env=settings.ENVIRONMENT)
    initialize_services()
    logger.info("app_ready")
    yield
    logger.info("app_shutdown")


# ── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="""
## AI Document Assistant API

Production-grade RAG pipeline with:
- 📄 Multi-format document ingestion (PDF, DOCX, TXT, MD, HTML)
- 🔍 FAISS/ChromaDB vector search with MMR re-ranking
- 🤖 GPT-4 / Ollama (open-source) answering
- 💬 Conversation memory with windowed history
- 📊 RAGAS evaluation metrics (answer relevance, faithfulness, precision, recall)
- 🌊 Server-Sent Events for streaming responses
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        response = Response("Internal error", status_code=500)
        try:
            response = await call_next(request)
        finally:
            elapsed = time.perf_counter() - start
            path = request.url.path
            REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
            REQUEST_LATENCY.labels(request.method, path).observe(elapsed)

            if path not in ("/metrics", settings.METRICS_PATH, "/health/live"):
                logger.info(
                    "http_request",
                    method=request.method,
                    path=path,
                    status=response.status_code,
                    latency_ms=round(elapsed * 1000, 1),
                )

        response.headers["X-Request-ID"] = request_id
        return response

    # ── Exception Handlers ────────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ── Routes ────────────────────────────────────────────────────────────

    prefix = settings.API_V1_PREFIX
    app.include_router(chat_router, prefix=prefix)
    app.include_router(docs_router, prefix=prefix)
    app.include_router(eval_router, prefix=prefix)
    app.include_router(health_router)

    @app.get(settings.METRICS_PATH, include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
