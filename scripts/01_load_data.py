"""Step 1: parse Professional.json and load into Postgres.

Run after `docker compose up -d`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import get_settings
from src.ingest.parse import collect_taxonomy, iter_records
from src.ingest.storage import bulk_insert_qa, init_schema, upsert_taxonomy


def main() -> None:
    s = get_settings()
    logger.info("loading {}", s.data_path)

    # Pass 1: collect taxonomy (needs full pass; data is ~27MB so fits in RAM easily)
    records = list(iter_records(s.data_path))
    logger.info("parsed {} qa records", len(records))

    specialties, sp_to_topics, chunks = collect_taxonomy(records)

    init_schema()
    upsert_taxonomy(specialties, sp_to_topics, chunks)

    n = bulk_insert_qa(records, batch_size=2000)
    logger.info("inserted/affected qa rows: {}", n)


if __name__ == "__main__":
    main()
