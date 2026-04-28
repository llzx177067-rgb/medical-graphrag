"""Fill summaries for topics that didn't get one in 06_build_topic_summaries.py.

Targets only topics where summary IS NULL (or empty). Embeds and upserts them.
Concurrency is intentionally small to coexist with other LLM jobs.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from src.clients.embedding import EmbeddingClient
from src.clients.llm import LLMClient
from src.index.vector_store import ensure_collections, upsert_topic_vectors
from src.ingest.prompts import TOPIC_SUMMARY_SYSTEM, build_topic_summary_user
from src.ingest.storage import pg_conn


def _fetch_missing(limit: int | None) -> list[dict]:
    sql = """
    WITH ranked AS (
        SELECT specialty, topic, question, answer,
               ROW_NUMBER() OVER (PARTITION BY specialty, topic ORDER BY part, seq_in_chunk) AS rn
          FROM qa_pair
    )
    SELECT t.specialty, t.name AS topic,
           jsonb_agg(jsonb_build_object('question', r.question, 'answer', r.answer)
                     ORDER BY r.rn) FILTER (WHERE r.rn <= 8) AS samples
      FROM topic t
      JOIN ranked r ON r.specialty = t.specialty AND r.topic = t.name
     WHERE (t.summary IS NULL OR length(t.summary) = 0)
     GROUP BY t.specialty, t.name
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
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
        text = ""
    return {"specialty": row["specialty"], "topic": row["topic"], "summary": (text or "").strip()}


async def main_async(concurrency: int, limit: int | None) -> None:
    rows = _fetch_missing(limit)
    if not rows:
        logger.info("no missing summaries; exiting")
        return
    logger.info("filling {} missing summaries (concurrency={})", len(rows), concurrency)

    llm = LLMClient(concurrency=concurrency)
    try:
        results = await tqdm_asyncio.gather(*[_summarise_one(llm, r) for r in rows], desc="fill-summary")
    finally:
        await llm.aclose()

    valid = [r for r in results if r.get("summary")]
    logger.info("ok={} skipped={}", len(valid), len(results) - len(valid))
    if not valid:
        return

    with pg_conn() as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE topic SET summary = %s WHERE specialty = %s AND name = %s",
            [(r["summary"], r["specialty"], r["topic"]) for r in valid],
        )
        conn.commit()

    async with EmbeddingClient() as emb:
        first = await emb.embed([valid[0]["summary"]])
        dim = len(first[0])
        ensure_collections(dim)
        STEP = 100
        for i in range(0, len(valid), STEP):
            chunk = valid[i : i + STEP]
            vecs = await emb.embed_batched([r["summary"] for r in chunk])
            upsert_topic_vectors(chunk, vecs)
    logger.info("filled {} topic summaries", len(valid))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    asyncio.run(main_async(args.concurrency, args.limit))


if __name__ == "__main__":
    main()
