"""
Evaluation service using RAGAS metrics.
Measures: answer_relevance, faithfulness, context_precision, context_recall.
Can run inline (per-query) or batch (dataset evaluation).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

from app.core.config import get_settings
from app.models.schemas import EvaluationReport, EvaluationScores

logger = structlog.get_logger(__name__)
settings = get_settings()


class EvaluationService:
    """
    Wraps RAGAS for RAG-specific metrics.
    Gracefully degrades if ragas is unavailable (returns None scores).
    """

    def __init__(self, llm: Any, embeddings: Any) -> None:
        self._llm = llm
        self._embeddings = embeddings
        self._ragas_available = self._check_ragas()

    # ── Public ────────────────────────────────────────────────────────────

    async def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationScores:
        """
        Evaluate a single QA pair. Returns scores in [0, 1].
        Falls back to heuristic scores if RAGAS is unavailable.
        """
        if not self._ragas_available:
            return self._heuristic_scores(question, answer, contexts)

        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._run_ragas_single(question, answer, contexts, ground_truth),
            )
        except Exception as exc:
            logger.warning("ragas_eval_failed", error=str(exc))
            return self._heuristic_scores(question, answer, contexts)

    async def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        namespace: str = "default",
    ) -> EvaluationReport:
        """Full batch evaluation — used by the /evaluate endpoint."""
        start = time.perf_counter()
        per_query: List[Dict[str, Any]] = []
        total_scores: Dict[str, List[float]] = {
            "answer_relevance": [],
            "faithfulness": [],
            "context_precision": [],
            "context_recall": [],
        }

        for i, (q, a, ctxs) in enumerate(zip(questions, answers, contexts_list)):
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            scores = await self.evaluate_single(q, a, ctxs, gt)
            per_query.append(
                {
                    "question": q,
                    "answer_preview": a[:200],
                    "scores": scores.model_dump(exclude_none=True),
                }
            )
            for key in total_scores:
                val = getattr(scores, key)
                if val is not None:
                    total_scores[key].append(val)

        def _avg(lst: List[float]) -> Optional[float]:
            return sum(lst) / len(lst) if lst else None

        aggregate = EvaluationScores(
            answer_relevance=_avg(total_scores["answer_relevance"]),
            faithfulness=_avg(total_scores["faithfulness"]),
            context_precision=_avg(total_scores["context_precision"]),
            context_recall=_avg(total_scores["context_recall"]),
        )

        elapsed = (time.perf_counter() - start) * 1000
        logger.info("batch_eval_complete", samples=len(questions), latency_ms=round(elapsed))

        return EvaluationReport(
            namespace=namespace,
            sample_size=len(questions),
            aggregate_scores=aggregate,
            per_query_scores=per_query,
            llm_model=settings.OPENAI_MODEL,
            retriever_config={
                "search_type": settings.RETRIEVER_SEARCH_TYPE,
                "top_k": settings.SIMILARITY_TOP_K,
                "reranking": settings.ENABLE_RERANKING,
            },
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _run_ragas_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str],
    ) -> EvaluationScores:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        metrics = [answer_relevancy, faithfulness, context_precision]

        if ground_truth:
            data["ground_truth"] = [ground_truth]
            metrics.append(context_recall)

        ds = Dataset.from_dict(data)
        ragas_llm = ChatOpenAI(
            model=settings.RAGAS_LLM_MODEL, api_key=settings.OPENAI_API_KEY
        )
        ragas_emb = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

        result = evaluate(
            dataset=ds,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_emb,
        )
        df = result.to_pandas()
        row = df.iloc[0]

        return EvaluationScores(
            answer_relevance=_safe_float(row.get("answer_relevancy")),
            faithfulness=_safe_float(row.get("faithfulness")),
            context_precision=_safe_float(row.get("context_precision")),
            context_recall=_safe_float(row.get("context_recall")) if ground_truth else None,
        )

    @staticmethod
    def _heuristic_scores(
        question: str, answer: str, contexts: List[str]
    ) -> EvaluationScores:
        """
        Fast heuristic fallback when RAGAS is unavailable.
        Uses simple keyword overlap as a proxy.
        """
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        ctx_text = " ".join(contexts).lower()
        ctx_words = set(ctx_text.split())

        # Relevance: overlap between answer and question
        relevance = len(q_words & a_words) / max(len(q_words), 1)

        # Faithfulness: answer words found in context
        faith_denom = max(len(a_words), 1)
        faithfulness = len(a_words & ctx_words) / faith_denom

        return EvaluationScores(
            answer_relevance=min(1.0, relevance * 5),  # scale up
            faithfulness=min(1.0, faithfulness),
            context_precision=None,
            context_recall=None,
        )

    @staticmethod
    def _check_ragas() -> bool:
        try:
            import ragas  # noqa: F401
            return True
        except ImportError:
            logger.info("ragas_not_installed_using_heuristics")
            return False


def _safe_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
        return f if 0.0 <= f <= 1.0 else None
    except (TypeError, ValueError):
        return None
