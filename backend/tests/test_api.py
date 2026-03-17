"""
Integration + unit tests for the RAG API.
Uses pytest-asyncio with httpx for async client testing.
"""
from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mock_settings(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "sk-test-key",
            "LLM_PROVIDER": "openai",
            "VECTOR_STORE_PATH": str(tmp / "vectors"),
            "UPLOAD_DIR": str(tmp / "uploads"),
            "ENVIRONMENT": "testing",
        },
    ):
        yield


@pytest.fixture(scope="session")
def app(mock_settings):
    """Build app with mocked LLM and embeddings."""
    with (
        patch("app.services.llm_factory.build_llm_and_embeddings") as mock_factory,
        patch("app.services.vector_store.VectorStoreService") as MockVS,
    ):
        mock_llm = MagicMock()
        mock_embeddings = MagicMock()
        mock_factory.return_value = (mock_llm, mock_embeddings)

        mock_vs_instance = MagicMock()
        mock_vs_instance.get_store_stats = AsyncMock(return_value={"total_vectors": 0})
        mock_vs_instance.add_documents = AsyncMock(return_value=["chunk-1"])
        mock_vs_instance.similarity_search = AsyncMock(return_value=[])
        mock_vs_instance.mmr_search = AsyncMock(return_value=[])
        MockVS.return_value = mock_vs_instance

        from app.main import create_app
        return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


# ── Health Tests ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_liveness(self, client):
        r = client.get("/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    def test_health_structure(self, client):
        r = client.get("/health/")
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            data = r.json()
            assert "version" in data
            assert "components" in data
            assert isinstance(data["components"], list)

    def test_root_endpoint(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "version" in r.json()


# ── Document Upload Tests ─────────────────────────────────────────────────────

class TestDocumentUpload:
    def test_upload_unsupported_format(self, client):
        r = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.xyz", b"content", "application/octet-stream")},
            data={"namespace": "test"},
        )
        assert r.status_code == 415

    def test_upload_empty_file_list(self, client):
        r = client.get("/api/v1/documents/")
        assert r.status_code == 200
        data = r.json()
        assert "documents" in data
        assert "total" in data

    def test_get_nonexistent_document(self, client):
        r = client.get("/api/v1/documents/nonexistent-id")
        assert r.status_code == 404

    def test_delete_nonexistent_document(self, client):
        r = client.delete("/api/v1/documents/nonexistent-id")
        assert r.status_code == 404


# ── Chat Tests ────────────────────────────────────────────────────────────────

class TestChat:
    @patch("app.api.chat.chat")
    def test_chat_request_validation(self, client):
        # Empty message should fail
        r = client.post(
            "/api/v1/chat/",
            json={"message": "", "namespace": "default"},
        )
        assert r.status_code == 422

    def test_chat_history_not_found(self, client):
        r = client.get("/api/v1/chat/nonexistent-session/history")
        assert r.status_code == 404

    def test_clear_nonexistent_session(self, client):
        r = client.delete("/api/v1/chat/nonexistent-session")
        assert r.status_code == 204


# ── RAG Pipeline Unit Tests ───────────────────────────────────────────────────

class TestRAGPipeline:
    def test_format_context(self):
        from langchain_core.documents import Document
        from app.services.rag_pipeline import RAGPipeline

        docs = [
            Document(page_content="Hello world", metadata={"filename": "test.pdf", "page": 1}),
            Document(page_content="Foo bar", metadata={"filename": "other.pdf"}),
        ]
        ctx = RAGPipeline._format_context(docs)
        assert "test.pdf" in ctx
        assert "Hello world" in ctx
        assert "Foo bar" in ctx

    def test_build_sources(self):
        from langchain_core.documents import Document
        from app.services.rag_pipeline import RAGPipeline

        docs = [
            Document(
                page_content="Sample content",
                metadata={
                    "document_id": "doc-1",
                    "filename": "sample.pdf",
                    "chunk_id": "abc123",
                    "page": 2,
                },
            )
        ]
        scores = [0.85]
        sources = RAGPipeline._build_sources(docs, scores)
        assert len(sources) == 1
        assert sources[0].document_id == "doc-1"
        assert sources[0].relevance_score == 0.85

    def test_build_sources_deduplication(self):
        from langchain_core.documents import Document
        from app.services.rag_pipeline import RAGPipeline

        docs = [
            Document(page_content="A", metadata={"chunk_id": "same", "document_id": "d1", "filename": "f.pdf"}),
            Document(page_content="B", metadata={"chunk_id": "same", "document_id": "d1", "filename": "f.pdf"}),
        ]
        sources = RAGPipeline._build_sources(docs, [0.9, 0.8])
        assert len(sources) == 1  # deduplicated


# ── Evaluation Unit Tests ─────────────────────────────────────────────────────

class TestEvaluation:
    def test_heuristic_scores_range(self):
        from app.services.evaluation import EvaluationService

        scores = EvaluationService._heuristic_scores(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."],
        )
        assert scores.answer_relevance is not None
        assert 0.0 <= scores.answer_relevance <= 1.0
        assert scores.faithfulness is not None
        assert 0.0 <= scores.faithfulness <= 1.0

    def test_heuristic_empty_context(self):
        from app.services.evaluation import EvaluationService

        scores = EvaluationService._heuristic_scores(
            question="What?", answer="I don't know", contexts=[]
        )
        assert scores.answer_relevance is not None


# ── Document Processor Unit Tests ─────────────────────────────────────────────

class TestDocumentProcessor:
    def test_chunk_id_deterministic(self):
        from app.services.document_processor import _make_chunk_id

        id1 = _make_chunk_id("doc-abc", 0)
        id2 = _make_chunk_id("doc-abc", 0)
        id3 = _make_chunk_id("doc-abc", 1)
        assert id1 == id2
        assert id1 != id3

    def test_unsupported_extension_raises(self):
        import asyncio
        from pathlib import Path
        from app.services.document_processor import DocumentProcessor

        mock_vs = MagicMock()
        processor = DocumentProcessor(vector_store=mock_vs)

        with pytest.raises(ValueError, match="Unsupported file type"):
            asyncio.get_event_loop().run_until_complete(
                processor._load(Path("test.xyz"))
            )


# ── Prometheus Metrics Tests ──────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_endpoint_accessible(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "http_requests_total" in r.text
