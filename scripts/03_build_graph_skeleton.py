"""Step 3: build the Specialty/Topic/Chunk/QAPair skeleton in Neo4j."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import get_settings
from src.graph.builder import upsert_skeleton
from src.graph.schema import init_schema as init_graph_schema
from src.ingest.parse import collect_taxonomy, iter_records


def main() -> None:
    s = get_settings()
    records = list(iter_records(s.data_path))
    logger.info("parsed {} qa records", len(records))

    specialties, sp_to_topics, chunks = collect_taxonomy(records)
    init_graph_schema()

    qa_iter = (
        {
            "qa_id": r.qa_id,
            "chunk_id": r.chunk_id,
            "specialty": r.specialty,
            "topic": r.topic,
            "seq_in_chunk": r.seq_in_chunk,
            "question": r.question,
            "answer": r.answer,
        }
        for r in records
    )
    upsert_skeleton(specialties, sp_to_topics, chunks, qa_iter, batch=1500)


if __name__ == "__main__":
    main()
