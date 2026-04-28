"""Step 2: build Q-vec and A-vec collections in Qdrant.

Uses DashScope embedding model (configured in .env).
Embeds Q and A independently, in parallel, with batch size from .env.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from loguru import logger
from tqdm import tqdm

from src.clients.embedding import EmbeddingClient
from src.config import get_settings
from src.index.vector_store import ensure_collections, existing_point_ids, upsert_qa_vectors, _qa_point_id
from src.ingest.storage import fetch_all_qa


def _truncate(text: str, n: int = 1500) -> str:
    return text if len(text) <= n else text[:n]


async def build(view: str, limit: int | None = None, resume: bool = True) -> None:
    s = get_settings()
    qa_rows = fetch_all_qa(limit=limit)
    if not qa_rows:
        logger.warning("no qa rows in postgres, run 01_load_data.py first")
        return

    if view == "q":
        target_collection = s.qdrant_collection_q
    elif view == "a":
        target_collection = s.qdrant_collection_a
    else:
        raise ValueError(view)

    # Resume: skip qa_ids whose point_id already exists in Qdrant
    if resume:
        try:
            already = existing_point_ids(target_collection)
            logger.info("resume: {} already in {}", len(already), target_collection)
            qa_rows = [r for r in qa_rows if _qa_point_id(r["qa_id"]) not in already]
        except Exception as e:
            logger.warning("could not resume from qdrant ({}); starting fresh", e)

    logger.info("embedding {} rows for view={}", len(qa_rows), view)
    if not qa_rows:
        logger.info("nothing to do for view={}", view)
        return

    if view == "q":
        texts = [r["question"] for r in qa_rows]
    else:
        texts = [_truncate(f"{r['question']}\n{r['answer']}") for r in qa_rows]

    async with EmbeddingClient() as emb:
        first = await emb.embed([texts[0]])
        dim = len(first[0])
        if dim != s.embed_dim:
            logger.warning("model returned dim={} but EMBED_DIM={}; using {}", dim, s.embed_dim, dim)
        ensure_collections(dim)

        BATCH = 2000  # smaller flush batch -> fewer Qdrant timeouts
        for i in tqdm(range(0, len(texts), BATCH), desc=f"embed-{view}"):
            sub_texts = texts[i : i + BATCH]
            sub_rows = qa_rows[i : i + BATCH]
            vectors = await emb.embed_batched(sub_texts)
            upsert_qa_vectors(target_collection, sub_rows, vectors)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--view", choices=["q", "a", "both"], default="both")
    p.add_argument("--limit", type=int, default=None, help="for smoke testing")
    args = p.parse_args()

    if args.view == "both":
        asyncio.run(build("q", args.limit))
        asyncio.run(build("a", args.limit))
    else:
        asyncio.run(build(args.view, args.limit))


if __name__ == "__main__":
    main()
