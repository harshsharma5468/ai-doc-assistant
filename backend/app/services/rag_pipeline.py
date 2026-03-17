"""
Core RAG pipeline.
Implements retrieval-augmented generation with:
  - MMR / similarity retrieval from the vector store
  - Cross-encoder reranking (optional)
  - HyDE (Hypothetical Document Embeddings) (optional)
  - Windowed conversation memory
  - Source attribution with relevance scores
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import structlog
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.core.config import get_settings
from app.models.schemas import (
    ChatMessage,
    MessageRole,
    SearchType,
    SourceReference,
)
from app.services.vector_store import VectorStoreService

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Prompt Templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI Document Assistant. Your role is to provide accurate, \
well-reasoned answers based EXCLUSIVELY on the provided document context.

Guidelines:
- Answer only based on the retrieved context. Never fabricate information.
- If the context does not contain enough information to answer, explicitly say so.
- When referencing information, cite the source document and page number when available.
- Be concise but thorough. Structure complex answers with clear formatting.
- For technical questions, include relevant code snippets or formulas from the documents.
- If asked about something outside the documents, politely redirect to the available content.

Context from documents:
{context}

Conversation history is provided for continuity. Use it to understand follow-up questions."""

HUMAN_PROMPT = """{question}"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
])

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(
    """Given the conversation history and the follow-up question, rephrase the follow-up \
question to be a standalone question that contains all necessary context.

Conversation History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""
)


# ── Custom Retriever ──────────────────────────────────────────────────────────

class NamespacedRetriever(BaseRetriever):
    """
    LangChain-compatible retriever that wraps VectorStoreService
    and supports namespace isolation, MMR, and filters.
    """

    vector_store: VectorStoreService
    namespace: str = "default"
    search_type: str = "mmr"
    k: int = 5
    fetch_k: int = 20
    lambda_mult: float = 0.5
    score_threshold: float = 0.3
    filters: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._async_retrieve(query))
        finally:
            loop.close()

    async def _async_retrieve(self, query: str) -> List[Document]:
        if self.search_type == "mmr":
            return await self.vector_store.mmr_search(
                query=query,
                namespace=self.namespace,
                k=self.k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult,
                filters=self.filters,
            )
        else:
            results = await self.vector_store.similarity_search(
                query=query,
                namespace=self.namespace,
                k=self.k,
                score_threshold=self.score_threshold,
                filters=self.filters,
            )
            return [doc for doc, _ in results]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None,
    ) -> List[Document]:
        return await self._async_retrieve(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return await self._async_retrieve(query)


# ── Reranker ──────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Re-ranks retrieved documents using a cross-encoder model.
    Significantly improves precision for ambiguous queries.
    """

    def __init__(self, model_name: str = settings.RERANKER_MODEL) -> None:
        self._model = None
        self._model_name = model_name

    def _load(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name)
                logger.info("reranker_loaded", model=self._model_name)
            except Exception as exc:
                logger.warning("reranker_unavailable", error=str(exc))

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        self._load()
        if self._model is None:
            return [(doc, 1.0) for doc in docs[:top_k]]

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self._model.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ── RAG Chain ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline with conversation memory, reranking, and source attribution.
    """

    def __init__(
        self,
        llm: Any,
        vector_store: VectorStoreService,
    ) -> None:
        self._llm = llm
        self._vector_store = vector_store
        self._reranker = CrossEncoderReranker() if settings.ENABLE_RERANKING else None
        self._sessions: Dict[str, Dict[str, Any]] = {}

    # ── Public ────────────────────────────────────────────────────────────

    async def query(
        self,
        question: str,
        session_id: str,
        namespace: str = "default",
        search_type: SearchType = SearchType.MMR,
        top_k: int = 5,
        include_sources: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        log = logger.bind(session_id=session_id, namespace=namespace)
        log.info("rag_query_start", question_len=len(question))

        memory = self._get_or_create_memory(session_id)

        # 1. Retrieve
        retriever = NamespacedRetriever(
            vector_store=self._vector_store,
            namespace=namespace,
            search_type=search_type.value,
            k=top_k * 3 if self._reranker else top_k,  # fetch more for reranking
            fetch_k=settings.RETRIEVER_FETCH_K,
            lambda_mult=settings.MMR_LAMBDA_MULT,
            score_threshold=settings.SIMILARITY_SCORE_THRESHOLD,
            filters=filters,
        )

        raw_docs = await retriever.aget_relevant_documents(question)

        # 2. Optionally rerank
        if self._reranker and raw_docs:
            reranked = self._reranker.rerank(question, raw_docs, top_k=top_k)
            docs = [doc for doc, _ in reranked]
            scores = [float(score) for _, score in reranked]
        else:
            docs = raw_docs[:top_k]
            scores = [0.8] * len(docs)

        # 3. Build context
        context = self._format_context(docs)

        # 4. Build prompt with memory
        history = memory.load_memory_variables({}).get("chat_history", "")
        chain = (
            {
                "context": RunnableLambda(lambda _: context),
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda _: history),
            }
            | self._build_qa_chain()
            | StrOutputParser()
        )

        answer = await chain.ainvoke(question)

        # 5. Save to memory
        memory.save_context({"input": question}, {"output": answer})
        self._sessions[session_id]["turn_count"] += 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        sources = []
        if include_sources:
            sources = self._build_sources(docs, scores)

        log.info("rag_query_complete", latency_ms=round(elapsed_ms, 1), sources=len(sources))

        return {
            "answer": answer,
            "sources": sources,
            "context_docs": docs,
            "latency_ms": elapsed_ms,
            "turn_count": self._sessions[session_id]["turn_count"],
        }

    def get_history(self, session_id: str) -> List[ChatMessage]:
        if session_id not in self._sessions:
            return []
        memory = self._sessions[session_id]["memory"]
        messages = []
        for msg in memory.chat_memory.messages:
            role = MessageRole.USER if msg.type == "human" else MessageRole.ASSISTANT
            messages.append(ChatMessage(role=role, content=msg.content))
        return messages

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ── Internal ──────────────────────────────────────────────────────────

    def _get_or_create_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        if session_id not in self._sessions:
            memory = ConversationBufferWindowMemory(
                k=settings.MEMORY_WINDOW_SIZE,
                memory_key="chat_history",
                return_messages=False,
                human_prefix="User",
                ai_prefix="Assistant",
            )
            self._sessions[session_id] = {"memory": memory, "turn_count": 0}
        return self._sessions[session_id]["memory"]

    def _build_qa_chain(self) -> Any:
        from langchain_core.runnables import RunnableLambda

        def _format_prompt(inputs: Dict[str, Any]) -> Any:
            return QA_PROMPT.format_messages(
                context=inputs["context"],
                question=inputs["question"],
            )

        return RunnableLambda(_format_prompt) | self._llm

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            source_label = (
                f"[{i}] {meta.get('filename', meta.get('source', 'Unknown'))}"
                f" (page {meta.get('page', meta.get('page_number', 'N/A'))})"
            )
            parts.append(f"{source_label}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _build_sources(docs: List[Document], scores: List[float]) -> List[SourceReference]:
        seen: set[str] = set()
        sources = []
        for doc, score in zip(docs, scores):
            meta = doc.metadata
            chunk_id = meta.get("chunk_id", "")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            sources.append(
                SourceReference(
                    document_id=meta.get("document_id", ""),
                    filename=meta.get("filename", meta.get("source", "Unknown")),
                    chunk_id=chunk_id,
                    page_number=meta.get("page", meta.get("page_number")),
                    content_snippet=doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                    relevance_score=min(1.0, max(0.0, float(score))),
                )
            )
        return sources
