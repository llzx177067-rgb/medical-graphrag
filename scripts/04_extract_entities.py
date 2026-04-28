"""Step 4: LLM-driven entity extraction over all QA pairs.

Output: data/cache/entities.jsonl  +  Postgres tables `entity`, `qa_entity_mention`.

Resumable: if entities.jsonl already has lines, those qa_ids are skipped.
Heavy step: with 30k QA at concurrency=8 expect ~30-60 minutes depending on model.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import get_settings
from src.ingest.entity_extract import extract_entities_async, normalise_and_persist
from src.ingest.storage import fetch_all_qa


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="extract only first N qa rows (smoke test)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--persist-only", action="store_true", help="skip extraction, just normalise & write to PG")
    args = p.parse_args()

    s = get_settings()
    cache_file = s.cache_path / "entities.jsonl"

    if not args.persist_only:
        qa_rows = fetch_all_qa(limit=args.limit)
        if not qa_rows:
            logger.error("no qa rows in postgres; run 01_load_data.py first")
            return
        asyncio.run(
            extract_entities_async(qa_rows, cache_path=cache_file, resume=not args.no_resume)
        )

    logger.info("normalising and persisting to postgres ...")
    n_ent, n_mention = normalise_and_persist(cache_file)
    logger.info("done: {} entities, {} mentions", n_ent, n_mention)


if __name__ == "__main__":
    main()
