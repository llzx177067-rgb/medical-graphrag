"""Resync topic summary vectors from Postgres → Qdrant.

Use when 06_build_topic_summaries.py finished the LLM + Postgres update phase
but failed during the final vector upsert (e.g. transient Qdrant disconnect).
This script does NOT call the LLM — it reads existing summaries straight from
Postgres and only does embed + upsert.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.clients.embedding import EmbeddingClient
from src.index.vector_store import ensure_collections, upsert_topic_vectors
from src.ingest.storage import pg_conn


async def main() -> None:
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT specialty, name AS topic, summary FROM topic "
            "WHERE summary IS NOT NULL AND length(summary) > 0"
        )
        rows = cur.fetchall()
    logger.info("resyncing {} topic summaries to qdrant", len(rows))
    if not rows:
        return

    async with EmbeddingClient() as emb:
        first = await emb.embed([rows[0]["summary"]])
        dim = len(first[0])
        ensure_collections(dim)
        # Embed in chunks of 200 to avoid bursts
        STEP = 200
        all_vecs: list[list[float]] = []
        for i in range(0, len(rows), STEP):
            chunk = rows[i : i + STEP]
            vecs = await emb.embed_batched([r["summary"] for r in chunk])
            all_vecs.extend(vecs)
            upsert_topic_vectors(chunk, vecs)
            logger.info("resynced {}/{}", min(i + STEP, len(rows)), len(rows))

    logger.info("done")


if __name__ == "__main__":
    asyncio.run(main())
