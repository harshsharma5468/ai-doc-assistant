"""
Pydantic domain models for the API layer.
All request/response schemas live here for a single source of truth.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SearchType(str, Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"
    HYBRID = "hybrid"


# ── Document Models ────────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    class Config:
        json_encoders = {bytes: lambda v: v.decode("utf-8")}


class DocumentRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: str
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: Optional[DocumentMetadata] = None
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    namespace: str = "default"

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    message: str
    chunk_count: Optional[int] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentRecord]
    total: int
    page: int
    page_size: int


# ── Chat / Conversation Models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceReference(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    page_number: Optional[int] = None
    content_snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    session_id: Optional[str] = None
    namespace: str = "default"
    search_type: SearchType = SearchType.MMR
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True
    stream: bool = False
    filters: Optional[Dict[str, Any]] = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceReference] = Field(default_factory=list)
    conversation_turn: int
    latency_ms: float
    model_used: str
    tokens_used: Optional[int] = None
    evaluation_scores: Optional[EvaluationScores] = None


class ConversationHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    document_namespace: str
    created_at: datetime
    last_activity: datetime
    turn_count: int


# ── Evaluation Models ──────────────────────────────────────────────────────────

class EvaluationScores(BaseModel):
    answer_relevance: Optional[float] = Field(None, ge=0.0, le=1.0)
    faithfulness: Optional[float] = Field(None, ge=0.0, le=1.0)
    context_precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    context_recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    answer_correctness: Optional[float] = Field(None, ge=0.0, le=1.0)


class EvaluationRequest(BaseModel):
    questions: List[str]
    ground_truths: Optional[List[str]] = None
    namespace: str = "default"
    sample_size: int = Field(default=10, ge=1, le=100)


class EvaluationReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    namespace: str
    sample_size: int
    aggregate_scores: EvaluationScores
    per_query_scores: List[Dict[str, Any]]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    llm_model: str
    retriever_config: Dict[str, Any]


# ── Health / Metrics Models ────────────────────────────────────────────────────

class ComponentHealth(BaseModel):
    name: str
    status: str  # "healthy" | "degraded" | "unhealthy"
    latency_ms: Optional[float] = None
    details: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    components: List[ComponentHealth]
    uptime_seconds: float


class MetricsSummary(BaseModel):
    total_documents: int
    total_conversations: int
    total_queries: int
    avg_latency_ms: float
    avg_answer_relevance: float
    avg_faithfulness: float
    queries_last_hour: int
    errors_last_hour: int
