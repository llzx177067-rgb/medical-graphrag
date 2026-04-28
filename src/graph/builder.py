"""Bulk-load taxonomy + QA pair nodes/edges into Neo4j."""
from __future__ import annotations

from typing import Iterable

from loguru import logger

from .client import session


def _topic_key(specialty: str, topic: str) -> str:
    return f"{specialty}::{topic}"


def upsert_skeleton(
    specialties: Iterable[str],
    sp_to_topics: dict[str, set[str]],
    chunks: dict[str, dict],
    qa_iter: Iterable[dict],
    *,
    batch: int = 1000,
) -> None:
    """Build Specialty/Topic/Chunk nodes plus BELONGS_TO edges, then QAPair nodes."""

    sp_list = list(specialties)
    topic_rows = [
        {"specialty": sp, "topic": t, "key": _topic_key(sp, t)}
        for sp, ts in sp_to_topics.items()
        for t in sorted(ts)
    ]
    chunk_rows = [
        {**meta, "topic_key": _topic_key(meta["specialty"], meta["topic"])}
        for meta in chunks.values()
    ]

    with session() as s:
        # Specialty
        s.run(
            """
            UNWIND $rows AS row
            MERGE (sp:Specialty {name: row})
            """,
            rows=sp_list,
        )

        # Topic + edge to specialty
        s.run(
            """
            UNWIND $rows AS row
            MERGE (t:Topic {key: row.key})
              ON CREATE SET t.name = row.topic, t.specialty = row.specialty
            WITH t, row
            MATCH (sp:Specialty {name: row.specialty})
            MERGE (t)-[:BELONGS_TO]->(sp)
            """,
            rows=topic_rows,
        )

        # Chunk + edge to topic
        for i in range(0, len(chunk_rows), batch):
            s.run(
                """
                UNWIND $rows AS row
                MERGE (c:Chunk {chunk_id: row.chunk_id})
                  ON CREATE SET c.specialty = row.specialty,
                                c.topic     = row.topic,
                                c.part      = row.part,
                                c.n_qa      = row.n_qa
                  ON MATCH  SET c.n_qa      = row.n_qa
                WITH c, row
                MATCH (t:Topic {key: row.topic_key})
                MERGE (c)-[:BELONGS_TO]->(t)
                """,
                rows=chunk_rows[i : i + batch],
            )

        # QAPair + edge to chunk
        qa_buf: list[dict] = []
        total = 0

        def _flush():
            nonlocal qa_buf, total
            if not qa_buf:
                return
            s.run(
                """
                UNWIND $rows AS row
                MERGE (q:QAPair {qa_id: row.qa_id})
                  ON CREATE SET q.question = row.question,
                                q.answer   = row.answer,
                                q.chunk_id = row.chunk_id,
                                q.specialty = row.specialty,
                                q.topic     = row.topic,
                                q.seq       = row.seq_in_chunk
                WITH q, row
                MATCH (c:Chunk {chunk_id: row.chunk_id})
                MERGE (q)-[:BELONGS_TO]->(c)
                """,
                rows=qa_buf,
            )
            total += len(qa_buf)
            qa_buf = []

        for r in qa_iter:
            qa_buf.append(
                {
                    "qa_id": r["qa_id"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "chunk_id": r["chunk_id"],
                    "specialty": r["specialty"],
                    "topic": r["topic"],
                    "seq_in_chunk": r["seq_in_chunk"],
                }
            )
            if len(qa_buf) >= batch:
                _flush()
        _flush()

        logger.info(
            "neo4j skeleton built: {} sp, {} topics, {} chunks, {} qa",
            len(sp_list),
            len(topic_rows),
            len(chunk_rows),
            total,
        )
