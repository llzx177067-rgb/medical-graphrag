"""FastAPI service exposing /ask and /retrieve, plus static frontend."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from ..clients.embedding import EmbeddingClient
from ..clients.llm import LLMClient
from ..clients.rerank import RerankClient
from ..generation.answer import answer_with_citations
from ..retrieval.pipeline import GraphRAGRetriever


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
    with_siblings: bool = True


class RetrieveRequest(BaseModel):
    question: str
    top_k: int | None = None


_state: dict[str, Any] = {}
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("warming up clients")
    _state["llm"] = LLMClient()
    _state["embedder"] = EmbeddingClient()
    _state["reranker"] = RerankClient()
    _state["retriever"] = GraphRAGRetriever(_state["llm"], _state["embedder"], _state["reranker"])
    try:
        yield
    finally:
        await _state["retriever"].aclose()


app = FastAPI(title="Medical GraphRAG", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/specialties")
async def specialties():
    """List specialties with their QA / topic counts."""
    from ..ingest.storage import pg_conn
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.name AS specialty,
                   s.n_topics,
                   s.n_qa,
                   COUNT(DISTINCT t.name) FILTER (WHERE t.summary IS NOT NULL AND length(t.summary)>0) AS topics_with_summary
              FROM specialty s
              LEFT JOIN topic t ON t.specialty = s.name
             GROUP BY s.name, s.n_topics, s.n_qa
             ORDER BY s.n_qa DESC
            """
        )
        return {"specialties": cur.fetchall()}


@app.get("/entity")
async def entity(name: str, limit: int = 10):
    """Inspect an entity: aliases, mention QAs, neighbors via typed relations."""
    from ..ingest.storage import pg_conn
    out: dict[str, Any] = {"name": name, "matches": [], "mentions": [], "relations": []}
    with pg_conn() as conn, conn.cursor() as cur:
        # Try exact name first, fall back to alias-contains
        cur.execute(
            """
            SELECT entity_id, name, type, aliases, description
              FROM entity
             WHERE name = %s OR %s = ANY(aliases)
             ORDER BY name
             LIMIT 5
            """,
            (name, name),
        )
        out["matches"] = cur.fetchall()
        if not out["matches"]:
            return out
        eids = [r["entity_id"] for r in out["matches"]]
        cur.execute(
            """
            SELECT q.qa_id, q.chunk_id, q.specialty, q.topic, q.question, q.answer, m.surface
              FROM qa_entity_mention m
              JOIN qa_pair q ON q.qa_id = m.qa_id
             WHERE m.entity_id = ANY(%s)
             ORDER BY q.specialty, q.topic
             LIMIT %s
            """,
            (eids, int(limit)),
        )
        out["mentions"] = cur.fetchall()
        cur.execute(
            """
            SELECT r.rel_type, e1.name AS head, e2.name AS tail,
                   r.confidence, r.evidence_chunk
              FROM relation r
              JOIN entity e1 ON e1.entity_id = r.head_id
              JOIN entity e2 ON e2.entity_id = r.tail_id
             WHERE r.head_id = ANY(%s) OR r.tail_id = ANY(%s)
             ORDER BY r.confidence DESC
             LIMIT %s
            """,
            (eids, eids, int(limit) * 3),
        )
        out["relations"] = cur.fetchall()
    return out


@app.get("/random_qa")
async def random_qa(specialty: str | None = None, n: int = 6):
    """Return random sample questions for inspiration."""
    from ..ingest.storage import pg_conn
    n = max(1, min(int(n), 20))
    sql = """
        SELECT question, answer, chunk_id, specialty, topic
          FROM qa_pair
         {where}
         ORDER BY random()
         LIMIT %s
    """
    where = ""
    params: list[Any] = []
    if specialty:
        where = "WHERE specialty = %s"
        params.append(specialty)
    params.append(n)
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql.format(where=where), tuple(params))
        return {"items": cur.fetchall()}


@app.get("/stats")
async def stats():
    """Live counts from Postgres / Neo4j / Qdrant."""
    out: dict[str, Any] = {"postgres": {}, "neo4j": {}, "qdrant": {}}

    # --- Postgres ---
    try:
        from ..ingest.storage import pg_conn
        with pg_conn() as conn, conn.cursor() as cur:
            for label, sql in [
                ("specialty",   "SELECT COUNT(*) AS c FROM specialty"),
                ("topic",       "SELECT COUNT(*) AS c FROM topic"),
                ("topic_with_summary",
                 "SELECT COUNT(*) AS c FROM topic WHERE summary IS NOT NULL AND length(summary) > 0"),
                ("chunk",       "SELECT COUNT(*) AS c FROM chunk"),
                ("qa_pair",     "SELECT COUNT(*) AS c FROM qa_pair"),
                ("entity",      "SELECT COUNT(*) AS c FROM entity"),
                ("qa_entity_mention",
                 "SELECT COUNT(*) AS c FROM qa_entity_mention"),
                ("relation",    "SELECT COUNT(*) AS c FROM relation"),
            ]:
                try:
                    cur.execute(sql)
                    out["postgres"][label] = cur.fetchone()["c"]
                except Exception as e:
                    out["postgres"][label] = f"error: {e}"
    except Exception as e:
        out["postgres"]["__error__"] = str(e)

    # --- Neo4j ---
    try:
        from ..graph.client import session as _s
        with _s() as gs:
            res = gs.run(
                """
                CALL apoc.meta.stats() YIELD labels, relTypesCount, nodeCount, relCount
                RETURN labels, relTypesCount, nodeCount, relCount
                """
            ).single()
            if res:
                out["neo4j"] = {
                    "node_count": res["nodeCount"],
                    "rel_count": res["relCount"],
                    "labels": res["labels"],
                    "rel_types": res["relTypesCount"],
                }
    except Exception as e:
        # Fallback if apoc.meta.stats unavailable
        try:
            from ..graph.client import session as _s
            with _s() as gs:
                row = gs.run("MATCH (n) RETURN count(n) AS n").single()
                rel = gs.run("MATCH ()-[r]->() RETURN count(r) AS r").single()
                out["neo4j"] = {"node_count": row["n"], "rel_count": rel["r"]}
        except Exception as e2:
            out["neo4j"]["__error__"] = str(e2)

    # --- Qdrant ---
    try:
        from ..index.vector_store import get_client
        from ..config import get_settings as _gs
        s = _gs()
        c = get_client()
        for name in (s.qdrant_collection_q, s.qdrant_collection_a, s.qdrant_collection_topic):
            try:
                info = c.get_collection(name)
                out["qdrant"][name] = {
                    "points": info.points_count,
                    "indexed": info.indexed_vectors_count,
                    "dim": info.config.params.vectors.size,
                }
            except Exception as e:
                out["qdrant"][name] = f"error: {e}"
    except Exception as e:
        out["qdrant"]["__error__"] = str(e)

    return out


def _bundle_payload(bundle, *, include_full_qa: bool = True) -> dict[str, Any]:
    """Common JSON shape for retrieval results."""
    items = []
    for it in bundle.items:
        row: dict[str, Any] = {
            "qa_id": it.qa_id,
            "score": it.score,
            "rrf_score": it.rrf_score,
            "chunk_id": it.chunk_id,
            "specialty": it.specialty,
            "topic": it.topic,
            "sources": it.sources,
            "channel_scores": it.channel_scores,
            "channel_ranks": it.channel_ranks,
        }
        if include_full_qa:
            row["question"] = it.question
            row["answer"] = it.answer
            row["siblings"] = it.siblings
        items.append(row)
    return {
        "query": bundle.query,
        "rewritten": bundle.rewritten,
        "entities": bundle.entities,
        "specialty_hint": bundle.specialty_hint,
        "items": items,
        "triples": bundle.triples,
        "timings_ms": bundle.timings_ms,
        "counts": bundle.counts,
    }


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    bundle = await _state["retriever"].retrieve(
        req.question, rerank_top_k=req.top_k, with_siblings=False
    )
    return _bundle_payload(bundle)


@app.post("/ask")
async def ask(req: AskRequest):
    bundle = await _state["retriever"].retrieve(
        req.question, rerank_top_k=req.top_k, with_siblings=req.with_siblings
    )
    answered = await answer_with_citations(_state["llm"], bundle)
    payload = _bundle_payload(bundle)
    payload["answer"] = answered.answer
    return payload


# ---- static frontend ------------------------------------------------------

if _STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")

    @app.get("/")
    async def root():
        return FileResponse(str(_STATIC_DIR / "index.html"))


def main():
    import uvicorn

    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
