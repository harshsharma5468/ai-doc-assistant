"""
Document ingestion pipeline.
Handles loading, splitting, embedding, and storing documents.
"""
from __future__ import annotations

import hashlib
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import structlog
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.models.schemas import (
    DocumentChunk,
    DocumentMetadata,
    DocumentRecord,
    DocumentStatus,
)
from app.services.vector_store import VectorStoreService

logger = structlog.get_logger(__name__)
settings = get_settings()


LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}


class DocumentProcessor:
    """
    Orchestrates the full document ingestion pipeline:
      1. Load raw content via loader registry
      2. Enrich metadata
      3. Split into semantically coherent chunks
      4. Embed chunks via the configured embedding model
      5. Persist to the vector store
    """

    def __init__(self, vector_store: VectorStoreService) -> None:
        self.vector_store = vector_store
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

    # ── Public Interface ───────────────────────────────────────────────────

    async def ingest(
        self,
        file_path: Path,
        document_id: str,
        namespace: str = "default",
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DocumentRecord, int]:
        """
        Full pipeline: load → chunk → embed → store.
        Returns (DocumentRecord, chunk_count).
        """
        start = time.perf_counter()
        log = logger.bind(document_id=document_id, path=str(file_path))
        log.info("ingestion_started")

        try:
            raw_docs = await self._load(file_path)
            log.info("loaded_raw_docs", count=len(raw_docs))

            metadata = self._extract_metadata(file_path, raw_docs)
            chunks = self._split(raw_docs, document_id, namespace, custom_metadata or {})
            log.info("chunks_created", count=len(chunks))

            await self.vector_store.add_documents(chunks, namespace=namespace)

            elapsed = (time.perf_counter() - start) * 1000
            log.info("ingestion_complete", chunk_count=len(chunks), latency_ms=round(elapsed, 1))

            record = DocumentRecord(
                id=document_id,  # type: ignore[arg-type]
                filename=file_path.name,
                status=DocumentStatus.INDEXED,
                metadata=metadata,
                chunk_count=len(chunks),
                namespace=namespace,
            )
            return record, len(chunks)

        except Exception as exc:
            log.exception("ingestion_failed", error=str(exc))
            raise

    async def delete_document(self, document_id: str, namespace: str = "default") -> int:
        """Remove all chunks belonging to a document from the vector store."""
        return await self.vector_store.delete_by_document_id(document_id, namespace)

    # ── Internal Steps ────────────────────────────────────────────────────

    async def _load(self, file_path: Path) -> List[Document]:
        suffix = file_path.suffix.lower()
        if suffix not in LOADER_MAP:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {list(LOADER_MAP.keys())}"
            )
        loader_cls = LOADER_MAP[suffix]
        loader = loader_cls(str(file_path))
        # Most LangChain loaders are synchronous; run in executor if needed
        return loader.load()

    def _split(
        self,
        docs: List[Document],
        document_id: str,
        namespace: str,
        custom_metadata: Dict[str, Any],
    ) -> List[Document]:
        chunks = self._splitter.split_documents(docs)
        for idx, chunk in enumerate(chunks):
            chunk_id = _make_chunk_id(document_id, idx)
            chunk.metadata.update(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "namespace": namespace,
                    **custom_metadata,
                }
            )
        return chunks

    def _extract_metadata(
        self, file_path: Path, docs: List[Document]
    ) -> DocumentMetadata:
        stat = file_path.stat()
        mime, _ = mimetypes.guess_type(str(file_path))
        word_count = sum(len(d.page_content.split()) for d in docs)
        page_count = max(
            (d.metadata.get("page", 0) for d in docs), default=0
        ) or len(docs)

        return DocumentMetadata(
            filename=file_path.name,
            file_type=mime or file_path.suffix,
            file_size_bytes=stat.st_size,
            page_count=page_count,
            word_count=word_count,
        )


def _make_chunk_id(document_id: str, index: int) -> str:
    raw = f"{document_id}:{index}"
    return hashlib.md5(raw.encode()).hexdigest()
