"""End-to-end hybrid retrieval pipeline.

Three recall channels (run concurrently):
  1. Q-vec   : query embedding vs. medgraph_q
  2. A-vec   : query embedding vs. medgraph_a
  3. Graph   : entity-anchored sub-graph traversal in Neo4j

Fusion: RRF, then DashScope rerank (cross-encoder), keep top-K.
Context assembly: hit QA + sibling QAs from same chunk + graph triples around anchors.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ..clients.embedding import EmbeddingClient
from ..clients.llm import LLMClient
from ..clients.rerank import RerankClient
from ..config import get_settings
from ..index.vector_store import vector_search
from ..ingest.storage import fetch_chunk_siblings, fetch_qa_by_ids
from .fusion import rrf_fuse
from .graph_recall import fetch_subgraph_triples, graph_recall
from .query_understanding import understand_query


@dataclass
class RetrievedItem:
    qa_id: str
    score: float           # final rerank score
    question: str
    answer: str
    chunk_id: str
    specialty: str
    topic: str
    sources: list[str] = field(default_factory=list)  # which channel(s) contributed
    channel_scores: dict[str, float] = field(default_factory=dict)  # raw similarity per channel
    channel_ranks: dict[str, int] = field(default_factory=dict)     # 1-based rank in each channel
    rrf_score: float = 0.0
    siblings: list[dict] = field(default_factory=list)


@dataclass
class RetrievalBundle:
    query: str
    rewritten: str
    entities: list[dict]
    specialty_hint: str
    items: list[RetrievedItem]
    triples: list[dict]
    timings_ms: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


class GraphRAGRetriever:
    def __init__(
        self,
        llm: LLMClient | None = None,
        embedder: EmbeddingClient | None = None,
        reranker: RerankClient | None = None,
    ):
        self._owns_clients = llm is None
        self.llm = llm or LLMClient()
        self.embedder = embedder or EmbeddingClient()
        self.reranker = reranker or RerankClient()
        self.s = get_settings()

    async def aclose(self):
        if self._owns_clients:
            await asyncio.gather(self.llm.aclose(), self.embedder.aclose(), self.reranker.aclose())

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    # ------------------------------------------------------------------
    async def retrieve(
        self,
        query: str,
        *,
        top_k_vec: int | None = None,
        top_k_graph: int | None = None,
        rerank_top_k: int | None = None,
        with_siblings: bool = True,
    ) -> RetrievalBundle:
        s = self.s
        top_k_vec = top_k_vec or s.recall_top_k_vector
        top_k_graph = top_k_graph or s.recall_top_k_graph
        rerank_top_k = rerank_top_k or s.rerank_top_k

        timings: dict[str, float] = {}
        T0 = time.perf_counter()

        # Stage 1: query understanding
        t = time.perf_counter()
        understanding = await understand_query(self.llm, query)
        timings["query_understanding"] = (time.perf_counter() - t) * 1000
        rewritten = understanding["rewritten"]
        ent_names = [e["name"] for e in understanding["entities"]]
        specialty_hint = understanding.get("specialty_hint") or None
        logger.info(
            "query understood: rewritten={!r} entities={} specialty={}",
            rewritten,
            ent_names,
            specialty_hint,
        )

        # Stage 2: parallel recall
        t = time.perf_counter()
        embed_task = asyncio.create_task(self.embedder.embed([rewritten]))

        async def _graph():
            return graph_recall(ent_names, top_k=top_k_graph) if ent_names else []

        graph_task = asyncio.create_task(_graph())

        qvec = (await embed_task)[0]
        timings["query_embed"] = (time.perf_counter() - t) * 1000

        def _v(coll):
            return vector_search(
                coll,
                qvec,
                top_k=top_k_vec,
                filter_specialty=specialty_hint,
            )

        t = time.perf_counter()
        v_q_task = asyncio.create_task(asyncio.to_thread(_v, s.qdrant_collection_q))
        v_a_task = asyncio.create_task(asyncio.to_thread(_v, s.qdrant_collection_a))

        graph_hits, vq_hits, va_hits = await asyncio.gather(graph_task, v_q_task, v_a_task)
        timings["recall"] = (time.perf_counter() - t) * 1000

        # If filtered vector search returns nothing, retry without filter
        if specialty_hint and not vq_hits and not va_hits:
            logger.info("specialty_hint filter pruned everything; retrying without filter")
            vq_hits = await asyncio.to_thread(vector_search, s.qdrant_collection_q, qvec, top_k=top_k_vec)
            va_hits = await asyncio.to_thread(vector_search, s.qdrant_collection_a, qvec, top_k=top_k_vec)

        # Build qa_id ranked lists + score/rank lookup tables
        list_q = [h.payload.get("qa_id") for h in vq_hits if h.payload.get("qa_id")]
        list_a = [h.payload.get("qa_id") for h in va_hits if h.payload.get("qa_id")]
        list_g = [h.qa_id for h in graph_hits]

        score_q = {h.payload.get("qa_id"): float(h.score) for h in vq_hits}
        score_a = {h.payload.get("qa_id"): float(h.score) for h in va_hits}
        score_g = {h.qa_id: float(h.score) for h in graph_hits}

        rank_q = {qid: i + 1 for i, qid in enumerate(list_q)}
        rank_a = {qid: i + 1 for i, qid in enumerate(list_a)}
        rank_g = {qid: i + 1 for i, qid in enumerate(list_g)}

        from_q = set(list_q)
        from_a = set(list_a)
        from_g = set(list_g)

        # Stage 3: RRF fusion
        fused = rrf_fuse([list_q, list_a, list_g], k=60, weights=[1.0, 1.0, 0.8])
        rrf_score_map = dict(fused)
        fused_ids = [qid for qid, _ in fused[: max(rerank_top_k * 4, 20)]]
        logger.info(
            "recall: q={} a={} graph={} fused={}",
            len(list_q), len(list_a), len(list_g), len(fused_ids),
        )

        # Stage 4: hydrate qa rows
        qa_rows = fetch_qa_by_ids(fused_ids)
        by_id = {r["qa_id"]: r for r in qa_rows}
        ordered = [by_id[i] for i in fused_ids if i in by_id]

        if not ordered:
            return RetrievalBundle(
                query=query,
                rewritten=rewritten,
                entities=understanding["entities"],
                specialty_hint=specialty_hint or "",
                items=[],
                triples=fetch_subgraph_triples(ent_names),
            )

        # Stage 5: cross-encoder rerank
        t = time.perf_counter()
        docs = [f"{r['question']}\n{r['answer']}" for r in ordered]
        try:
            ranked = await self.reranker.rerank(rewritten, docs, top_n=rerank_top_k)
        except Exception as exc:
            logger.warning("rerank failed, falling back to RRF order: {}", exc)
            ranked = [(i, 1.0 / (i + 1)) for i in range(min(rerank_top_k, len(ordered)))]
        timings["rerank"] = (time.perf_counter() - t) * 1000

        items: list[RetrievedItem] = []
        for idx, score in ranked:
            r = ordered[idx]
            qid = r["qa_id"]
            sources = []
            if qid in from_q: sources.append("q")
            if qid in from_a: sources.append("a")
            if qid in from_g: sources.append("graph")
            channel_scores = {}
            channel_ranks = {}
            if qid in score_q:
                channel_scores["q"] = score_q[qid]; channel_ranks["q"] = rank_q[qid]
            if qid in score_a:
                channel_scores["a"] = score_a[qid]; channel_ranks["a"] = rank_a[qid]
            if qid in score_g:
                channel_scores["graph"] = score_g[qid]; channel_ranks["graph"] = rank_g[qid]
            siblings = fetch_chunk_siblings(qid) if with_siblings else []
            items.append(
                RetrievedItem(
                    qa_id=qid,
                    score=float(score),
                    question=r["question"],
                    answer=r["answer"],
                    chunk_id=r["chunk_id"],
                    specialty=r["specialty"],
                    topic=r["topic"],
                    sources=sources,
                    channel_scores=channel_scores,
                    channel_ranks=channel_ranks,
                    rrf_score=float(rrf_score_map.get(qid, 0.0)),
                    siblings=siblings[:3],
                )
            )

        t = time.perf_counter()
        triples = fetch_subgraph_triples(ent_names) if ent_names else []
        timings["triples"] = (time.perf_counter() - t) * 1000

        timings["total"] = (time.perf_counter() - T0) * 1000

        counts = {
            "vec_q": len(list_q),
            "vec_a": len(list_a),
            "graph": len(list_g),
            "fused": len(fused_ids),
            "rerank_kept": len(items),
            "triples": len(triples),
        }

        return RetrievalBundle(
            query=query,
            rewritten=rewritten,
            entities=understanding["entities"],
            specialty_hint=specialty_hint or "",
            items=items,
            triples=triples,
            timings_ms=timings,
            counts=counts,
        )
