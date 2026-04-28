"""Step 5: LLM relation extraction over QA pairs that have >= 2 entities,
then push entities + relations into Neo4j.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import get_settings
from src.ingest.relation_extract import (
    extract_relations_async,
    fetch_qa_with_entities,
    persist_relations_to_pg,
    push_graph_layer,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--persist-only", action="store_true")
    p.add_argument("--push-graph", action="store_true",
                   help="after persisting to PG, push entities/relations into Neo4j")
    args = p.parse_args()

    s = get_settings()
    cache_file = s.cache_path / "relations.jsonl"

    if not args.persist_only:
        qa_rows = fetch_qa_with_entities(min_entities=2, limit=args.limit)
        if not qa_rows:
            logger.error("no qa pairs with >= 2 entities; run 04_extract_entities.py first")
            return
        asyncio.run(
            extract_relations_async(qa_rows, cache_path=cache_file, resume=not args.no_resume)
        )

    persist_relations_to_pg(cache_file)

    if args.push_graph:
        logger.info("pushing entities + relations into neo4j ...")
        push_graph_layer()


if __name__ == "__main__":
    main()
