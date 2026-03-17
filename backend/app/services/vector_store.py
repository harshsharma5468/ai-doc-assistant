"""
Vector store abstraction layer.
Supports FAISS (local) and ChromaDB (server) with a unified interface.
MMR retrieval, filtered search, and namespace isolation are first-class.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
from langchain_core.documents import Document
from langchain.vectorstores.base import VectorStore

from app.core.config import VectorStoreType, get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class VectorStoreService:
    """
    Thin async wrapper around LangChain vector stores.
    Uses namespace-keyed stores so that document sets are isolated.
    """

    def __init__(self, embeddings: Any) -> None:
        self._embeddings = embeddings
        self._stores: Dict[str, VectorStore] = {}
        self._lock = asyncio.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def get_or_create_store(self, namespace: str) -> VectorStore:
        if namespace not in self._stores:
            async with self._lock:
                if namespace not in self._stores:
                    self._stores[namespace] = await self._create_store(namespace)
        return self._stores[namespace]

    async def _create_store(self, namespace: str) -> VectorStore:
        store_type = settings.VECTOR_STORE_TYPE
        log = logger.bind(namespace=namespace, store_type=store_type)

        if store_type == VectorStoreType.FAISS:
            store = await self._load_or_init_faiss(namespace)
        elif store_type == VectorStoreType.CHROMA:
            store = self._init_chroma(namespace)
        else:
            raise ValueError(f"Unsupported vector store: {store_type}")

        log.info("vector_store_ready")
        return store

    # ── FAISS ─────────────────────────────────────────────────────────────

    async def _load_or_init_faiss(self, namespace: str) -> VectorStore:
        from langchain_community.vectorstores import FAISS

        store_path = settings.VECTOR_STORE_PATH / namespace
        if store_path.exists() and (store_path / "index.faiss").exists():
            logger.info("loading_existing_faiss", path=str(store_path))
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: FAISS.load_local(
                    str(store_path),
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                ),
            )

        logger.info("initializing_new_faiss", path=str(store_path))
        # Seed with a dummy doc to create the index structure
        store = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: FAISS.from_texts(
                ["__init__"],
                self._embeddings,
                metadatas=[{"__init__": True}],
            ),
        )
        store_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(store_path))
        return store

    async def _persist_faiss(self, namespace: str) -> None:
        store = self._stores.get(namespace)
        if store and settings.VECTOR_STORE_TYPE == VectorStoreType.FAISS:
            store_path = settings.VECTOR_STORE_PATH / namespace
            store_path.mkdir(parents=True, exist_ok=True)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: store.save_local(str(store_path))
            )

    # ── Chroma ────────────────────────────────────────────────────────────

    def _init_chroma(self, namespace: str) -> VectorStore:
        import chromadb
        from langchain_community.vectorstores import Chroma

        client = chromadb.HttpClient(
            host=settings.CHROMA_HOST, port=settings.CHROMA_PORT
        )
        return Chroma(
            collection_name=namespace,
            embedding_function=self._embeddings,
            client=client,
        )

    # ── Core Operations ───────────────────────────────────────────────────

    async def add_documents(
        self, docs: List[Document], namespace: str = "default"
    ) -> List[str]:
        store = await self.get_or_create_store(namespace)
        loop = asyncio.get_event_loop()
        ids = await loop.run_in_executor(None, lambda: store.add_documents(docs))
        await self._persist_faiss(namespace)
        logger.info("docs_added", count=len(docs), namespace=namespace)
        return ids or []

    async def similarity_search(
        self,
        query: str,
        namespace: str = "default",
        k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        store = await self.get_or_create_store(namespace)
        loop = asyncio.get_event_loop()

        kwargs: Dict[str, Any] = {"k": k}
        if filters:
            kwargs["filter"] = filters

        results: List[Tuple[Document, float]] = await loop.run_in_executor(
            None,
            lambda: store.similarity_search_with_relevance_scores(query, **kwargs),
        )
        filtered = [(doc, score) for doc, score in results if score >= score_threshold]
        logger.debug("similarity_search", query_len=len(query), results=len(filtered))
        return filtered

    async def mmr_search(
        self,
        query: str,
        namespace: str = "default",
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        store = await self.get_or_create_store(namespace)
        loop = asyncio.get_event_loop()

        kwargs: Dict[str, Any] = {
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        }
        if filters:
            kwargs["filter"] = filters

        results: List[Document] = await loop.run_in_executor(
            None,
            lambda: store.max_marginal_relevance_search(query, **kwargs),
        )
        logger.debug("mmr_search", query_len=len(query), results=len(results))
        return results

    async def delete_by_document_id(
        self, document_id: str, namespace: str = "default"
    ) -> int:
        store = await self.get_or_create_store(namespace)
        loop = asyncio.get_event_loop()

        # Retrieve IDs of all chunks belonging to this document
        results = await self.similarity_search(
            query="",  # empty query to get all
            namespace=namespace,
            k=10000,
            filters={"document_id": document_id},
        )
        chunk_ids = [
            doc.metadata.get("chunk_id") for doc, _ in results if doc.metadata.get("chunk_id")
        ]

        if chunk_ids:
            try:
                await loop.run_in_executor(None, lambda: store.delete(chunk_ids))
                await self._persist_faiss(namespace)
            except Exception as exc:
                logger.warning("delete_partial_failure", error=str(exc))

        logger.info("deleted_chunks", document_id=document_id, count=len(chunk_ids))
        return len(chunk_ids)

    async def get_store_stats(self, namespace: str = "default") -> Dict[str, Any]:
        store = await self.get_or_create_store(namespace)
        try:
            if hasattr(store, "index"):  # FAISS
                count = store.index.ntotal
            elif hasattr(store, "_collection"):  # Chroma
                count = store._collection.count()
            else:
                count = -1
        except Exception:
            count = -1
        return {"namespace": namespace, "total_vectors": count}
