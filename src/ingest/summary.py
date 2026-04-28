"""LLM-based topic summary generation.

For each (specialty, topic) we sample up to 8 representative QA pairs and ask
the LLM to write a 100-180 char Chinese mini-overview. Summaries are stored:
- Postgres `topic.summary`
- Qdrant `medgraph_topic` collection (with embedding)
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from ..clients.embedding import EmbeddingClient
from ..clients.llm import LLMClient
from ..config import get_settings
from ..index.vector_store import ensure_collections, upsert_topic_vectors
from .prompts import TOPIC_SUMMARY_SYSTEM, build_topic_summary_user
from .storage import pg_conn


def _fetch_topics_with_qa(per_topic: int = 8) -> list[dict]:
    sql = """
    WITH ranked AS (
        SELECT specialty, topic, question, answer,
               ROW_NUMBER() OVER (PARTITION BY specialty, topic ORDER BY part, seq_in_chunk) AS rn
          FROM qa_pair
    )
    SELECT specialty, topic,
           jsonb_agg(jsonb_build_object('question', question, 'answer', answer)
                     ORDER BY rn) FILTER (WHERE rn <= %s) AS samples
      FROM ranked
     GROUP BY specialty, topic
    """
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (per_topic,))
        return cur.fetchall()


async def _summarise_one(llm: LLMClient, row: dict) -> dict:
    user = build_topic_summary_user(row["specialty"], row["topic"], row["samples"] or [])
    try:
        text = await llm.chat(
            messages=[
                {"role": "system", "content": TOPIC_SUMMARY_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=400,
        )
    except Exception as exc:
        logger.warning("summary failed for {}::{}: {}", row["specialty"], row["topic"], exc)
        text = ""  # leave empty; caller will skip empty summaries
    return {"specialty": row["specialty"], "topic": row["topic"], "summary": (text or "").strip()}


async def build_topic_summaries(*, limit: int | None = None, concurrency: int | None = None) -> list[dict]:
    rows = _fetch_topics_with_qa()
    if limit:
        rows = rows[:limit]
    logger.info("summarising {} topics", len(rows))

    # Allow caller (or env) to override default concurrency for this lighter task.
    kw = {"concurrency": concurrency} if concurrency else {}
    llm = LLMClient(**kw)
    try:
        results = await tqdm_asyncio.gather(
            *[_summarise_one(llm, r) for r in rows], desc="topic-summary"
        )
    finally:
        await llm.aclose()

    # Persist to Postgres (only non-empty summaries)
    valid = [r for r in results if r.get("summary")]
    logger.info("summaries: total={} ok={} skipped={}", len(results), len(valid), len(results) - len(valid))
    if valid:
        with pg_conn() as conn, conn.cursor() as cur:
            cur.executemany(
                "UPDATE topic SET summary = %s WHERE specialty = %s AND name = %s",
                [(r["summary"], r["specialty"], r["topic"]) for r in valid],
            )
            conn.commit()

    # Embed and push to Qdrant medgraph_topic (only valid summaries)
    if valid:
        async with EmbeddingClient() as emb:
            first = await emb.embed([valid[0]["summary"]])
            dim = len(first[0])
            ensure_collections(dim)
            vectors = await emb.embed_batched([r["summary"] for r in valid])
        upsert_topic_vectors(valid, vectors)
    logger.info("topic summaries persisted ({} topics)", len(valid))
    return results
