"""
Document management endpoints.
Upload, list, delete, and inspect documents.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile, status

from app.core.config import get_settings
from app.core.dependencies import DocProcessorDep, VectorStoreDep
from app.models.schemas import (
    DocumentListResponse,
    DocumentRecord,
    DocumentStatus,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = structlog.get_logger(__name__)
settings = get_settings()

# In-memory document registry (replace with Redis/DB in full production)
_documents: dict[str, DocumentRecord] = {}


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and index a document",
)
async def upload_document(
    background_tasks: BackgroundTasks,
    doc_processor: DocProcessorDep,
    file: UploadFile = File(..., description="PDF, DOCX, TXT, MD, or HTML"),
    namespace: str = Form(default="default", description="Collection namespace"),
    metadata: Optional[str] = Form(default=None, description="JSON metadata string"),
):
    """
    Upload a document for RAG indexing.
    Processing is async — poll GET /documents/{id} for status.
    """
    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported type {suffix!r}. Allowed: {settings.SUPPORTED_EXTENSIONS}",
        )

    # Validate size
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max: {settings.MAX_UPLOAD_SIZE_MB} MB",
        )

    document_id = str(uuid.uuid4())
    dest_path = settings.UPLOAD_DIR / f"{document_id}{suffix}"
    dest_path.write_bytes(content)

    record = DocumentRecord(
        id=document_id,  # type: ignore[arg-type]
        filename=file.filename or "unknown",
        status=DocumentStatus.PENDING,
        namespace=namespace,
    )
    _documents[document_id] = record
    logger.info("document_uploaded", document_id=document_id, filename=file.filename)

    background_tasks.add_task(
        _index_document, document_id, dest_path, namespace, doc_processor
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename or "",
        status=DocumentStatus.PENDING,
        message="Document accepted for indexing. Check status via GET /documents/{id}",
    )


@router.get(
    "/{document_id}",
    response_model=DocumentRecord,
    summary="Get document status",
)
async def get_document(document_id: str):
    record = _documents.get(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return record


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all documents",
)
async def list_documents(
    namespace: Optional[str] = Query(default=None),
    status: Optional[DocumentStatus] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    docs = list(_documents.values())
    if namespace:
        docs = [d for d in docs if d.namespace == namespace]
    if status:
        docs = [d for d in docs if d.status == status]

    total = len(docs)
    start = (page - 1) * page_size
    paginated = docs[start : start + page_size]

    return DocumentListResponse(
        documents=paginated,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
)
async def delete_document(
    document_id: str,
    doc_processor: DocProcessorDep,
):
    record = _documents.get(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    await doc_processor.delete_document(document_id, namespace=record.namespace)

    # Clean up file
    for suffix in settings.SUPPORTED_EXTENSIONS:
        path = settings.UPLOAD_DIR / f"{document_id}{suffix}"
        if path.exists():
            path.unlink()

    del _documents[document_id]
    logger.info("document_deleted", document_id=document_id)


@router.get(
    "/{document_id}/stats",
    summary="Get vector store stats for a document namespace",
)
async def document_stats(document_id: str, vector_store: VectorStoreDep):
    record = _documents.get(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    stats = await vector_store.get_store_stats(record.namespace)
    return {"document_id": document_id, "record": record, "vector_store_stats": stats}


# ── Background Tasks ──────────────────────────────────────────────────────────

async def _index_document(
    document_id: str,
    file_path: Path,
    namespace: str,
    doc_processor: "DocumentProcessor",  # noqa: F821
) -> None:
    record = _documents.get(document_id)
    if not record:
        return

    _documents[document_id] = record.model_copy(update={"status": DocumentStatus.PROCESSING})
    try:
        updated_record, chunk_count = await doc_processor.ingest(
            file_path=file_path,
            document_id=document_id,
            namespace=namespace,
        )
        _documents[document_id] = updated_record
        logger.info("document_indexed", document_id=document_id, chunks=chunk_count)
    except Exception as exc:
        logger.exception("document_indexing_failed", document_id=document_id, error=str(exc))
        _documents[document_id] = record.model_copy(
            update={"status": DocumentStatus.FAILED, "error_message": str(exc)}
        )
