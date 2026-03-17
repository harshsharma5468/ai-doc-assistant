"""
Chat / RAG query endpoints.
Supports standard JSON responses and Server-Sent Events for streaming.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Optional

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.dependencies import EvalServiceDep, RAGPipelineDep
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    SearchType,
)

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = structlog.get_logger(__name__)
settings = get_settings()


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Ask a question with RAG",
    description="Sends a question through the RAG pipeline and returns an answer with source citations.",
)
async def chat(
    request: ChatRequest,
    rag: RAGPipelineDep,
    eval_service: EvalServiceDep,
):
    session_id = request.session_id or str(uuid.uuid4())

    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use POST /chat/stream for streaming responses.",
        )

    result = await rag.query(
        question=request.message,
        session_id=session_id,
        namespace=request.namespace,
        search_type=request.search_type,
        top_k=request.top_k,
        include_sources=request.include_sources,
        filters=request.filters,
    )

    # Async evaluation (non-blocking)
    eval_scores = None
    if settings.ENABLE_TRACING and result["context_docs"]:
        try:
            contexts = [doc.page_content for doc in result["context_docs"]]
            eval_scores = await eval_service.evaluate_single(
                question=request.message,
                answer=result["answer"],
                contexts=contexts,
            )
        except Exception as exc:
            logger.warning("inline_eval_failed", error=str(exc))

    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        sources=result["sources"],
        conversation_turn=result["turn_count"],
        latency_ms=round(result["latency_ms"], 1),
        model_used=settings.OPENAI_MODEL,
        evaluation_scores=eval_scores,
    )


@router.post(
    "/stream",
    summary="Stream chat response via SSE",
    response_class=StreamingResponse,
)
async def chat_stream(
    request: ChatRequest,
    rag: RAGPipelineDep,
):
    """
    Server-Sent Events endpoint.
    Emits: data: {"type": "token", "content": "..."} per token,
           data: {"type": "sources", "sources": [...]} at end,
           data: {"type": "done"} to terminate.
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def generate() -> AsyncGenerator[str, None]:
        # First emit session_id
        yield _sse_event({"type": "session", "session_id": session_id})

        # Run RAG (non-streaming retrieval + streaming generation)
        result = await rag.query(
            question=request.message,
            session_id=session_id,
            namespace=request.namespace,
            search_type=request.search_type,
            top_k=request.top_k,
            include_sources=request.include_sources,
            filters=request.filters,
        )

        # Simulate token streaming from full answer
        answer = result["answer"]
        words = answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield _sse_event({"type": "token", "content": chunk})
            await asyncio.sleep(0.02)  # ~50 tokens/sec pace

        # Emit sources
        sources_data = [s.model_dump() for s in result["sources"]]
        yield _sse_event({
            "type": "sources",
            "sources": sources_data,
            "turn": result["turn_count"],
        })
        yield _sse_event({"type": "done"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/{session_id}/history",
    response_model=ConversationHistory,
    summary="Get conversation history",
)
async def get_history(session_id: str, rag: RAGPipelineDep):
    messages = rag.get_history(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found or empty")

    return ConversationHistory(
        session_id=session_id,
        messages=messages,
        document_namespace="default",
        created_at=messages[0].timestamp if messages else __import__("datetime").datetime.utcnow(),
        last_activity=messages[-1].timestamp if messages else __import__("datetime").datetime.utcnow(),
        turn_count=len([m for m in messages if m.role.value == "user"]),
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear a conversation session",
)
async def clear_session(session_id: str, rag: RAGPipelineDep):
    rag.clear_session(session_id)
    logger.info("session_cleared", session_id=session_id)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
