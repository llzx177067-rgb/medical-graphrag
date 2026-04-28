"""Postgres storage for QA pairs and taxonomy.

Uses psycopg3 sync API — Postgres I/O is fast enough that an async path adds
no value here, and most ingestion scripts run as CLIs.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

import psycopg
from loguru import logger
from psycopg.rows import dict_row

from ..config import get_settings
from .parse import QARecord


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS specialty (
    name        TEXT PRIMARY KEY,
    n_topics    INTEGER NOT NULL DEFAULT 0,
    n_qa        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS topic (
    specialty   TEXT NOT NULL REFERENCES specialty(name) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    n_qa        INTEGER NOT NULL DEFAULT 0,
    summary     TEXT,
    PRIMARY KEY (specialty, name)
);

CREATE TABLE IF NOT EXISTS chunk (
    chunk_id    TEXT PRIMARY KEY,
    specialty   TEXT NOT NULL,
    topic       TEXT NOT NULL,
    part        INTEGER NOT NULL,
    n_qa        INTEGER NOT NULL DEFAULT 0,
    summary     TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunk_specialty ON chunk(specialty);
CREATE INDEX IF NOT EXISTS idx_chunk_topic     ON chunk(specialty, topic);

CREATE TABLE IF NOT EXISTS qa_pair (
    qa_id       TEXT PRIMARY KEY,
    chunk_id    TEXT NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    specialty   TEXT NOT NULL,
    topic       TEXT NOT NULL,
    part        INTEGER NOT NULL,
    seq_in_chunk INTEGER NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    qa_text     TEXT GENERATED ALWAYS AS (question || E'\n' || answer) STORED
);
CREATE INDEX IF NOT EXISTS idx_qa_chunk     ON qa_pair(chunk_id);
CREATE INDEX IF NOT EXISTS idx_qa_specialty ON qa_pair(specialty);
CREATE INDEX IF NOT EXISTS idx_qa_topic     ON qa_pair(specialty, topic);

CREATE TABLE IF NOT EXISTS entity (
    entity_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,
    aliases     TEXT[] NOT NULL DEFAULT '{}',
    description TEXT
);
CREATE INDEX IF NOT EXISTS idx_entity_name_type ON entity(name, type);
CREATE INDEX IF NOT EXISTS idx_entity_type ON entity(type);

CREATE TABLE IF NOT EXISTS qa_entity_mention (
    qa_id      TEXT NOT NULL REFERENCES qa_pair(qa_id) ON DELETE CASCADE,
    entity_id  TEXT NOT NULL REFERENCES entity(entity_id) ON DELETE CASCADE,
    surface    TEXT NOT NULL,
    PRIMARY KEY (qa_id, entity_id, surface)
);
CREATE INDEX IF NOT EXISTS idx_mention_entity ON qa_entity_mention(entity_id);

CREATE TABLE IF NOT EXISTS relation (
    rel_id      BIGSERIAL PRIMARY KEY,
    head_id     TEXT NOT NULL REFERENCES entity(entity_id) ON DELETE CASCADE,
    rel_type    TEXT NOT NULL,
    tail_id     TEXT NOT NULL REFERENCES entity(entity_id) ON DELETE CASCADE,
    evidence_qa TEXT NOT NULL,           -- qa_id
    evidence_chunk TEXT NOT NULL,        -- chunk_id
    confidence  REAL NOT NULL DEFAULT 0.7
);
CREATE INDEX IF NOT EXISTS idx_relation_head ON relation(head_id, rel_type);
CREATE INDEX IF NOT EXISTS idx_relation_tail ON relation(tail_id, rel_type);
CREATE UNIQUE INDEX IF NOT EXISTS uq_relation
    ON relation(head_id, rel_type, tail_id, evidence_qa);
"""


@contextmanager
def pg_conn() -> Iterator[psycopg.Connection]:
    s = get_settings()
    with psycopg.connect(s.postgres_dsn, autocommit=False, row_factory=dict_row) as conn:
        yield conn


def init_schema() -> None:
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        conn.commit()
    logger.info("postgres schema ready")


def upsert_taxonomy(
    specialties: Iterable[str],
    sp_to_topics: dict[str, set[str]],
    chunks: dict[str, dict],
) -> None:
    sp_list = list(specialties)
    with pg_conn() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO specialty(name, n_topics, n_qa)
            VALUES (%s, %s, 0)
            ON CONFLICT (name) DO UPDATE SET n_topics = EXCLUDED.n_topics
            """,
            [(s, len(sp_to_topics.get(s, set()))) for s in sp_list],
        )
        topic_rows = [
            (sp, t)
            for sp, ts in sp_to_topics.items()
            for t in sorted(ts)
        ]
        cur.executemany(
            """
            INSERT INTO topic(specialty, name)
            VALUES (%s, %s)
            ON CONFLICT (specialty, name) DO NOTHING
            """,
            topic_rows,
        )
        cur.executemany(
            """
            INSERT INTO chunk(chunk_id, specialty, topic, part, n_qa)
            VALUES (%(chunk_id)s, %(specialty)s, %(topic)s, %(part)s, %(n_qa)s)
            ON CONFLICT (chunk_id) DO UPDATE SET n_qa = EXCLUDED.n_qa
            """,
            list(chunks.values()),
        )
        conn.commit()
    logger.info(
        "taxonomy upserted: {} specialties, {} topics, {} chunks",
        len(sp_list),
        sum(len(v) for v in sp_to_topics.values()),
        len(chunks),
    )


def bulk_insert_qa(records: Iterable[QARecord], batch_size: int = 1000) -> int:
    rows: list[tuple] = []
    inserted = 0
    with pg_conn() as conn, conn.cursor() as cur:
        for r in records:
            rows.append(
                (r.qa_id, r.chunk_id, r.specialty, r.topic, r.part, r.seq_in_chunk, r.question, r.answer)
            )
            if len(rows) >= batch_size:
                cur.executemany(
                    """
                    INSERT INTO qa_pair(qa_id, chunk_id, specialty, topic, part, seq_in_chunk, question, answer)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (qa_id) DO NOTHING
                    """,
                    rows,
                )
                inserted += cur.rowcount
                rows.clear()
                conn.commit()
        if rows:
            cur.executemany(
                """
                INSERT INTO qa_pair(qa_id, chunk_id, specialty, topic, part, seq_in_chunk, question, answer)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (qa_id) DO NOTHING
                """,
                rows,
            )
            inserted += cur.rowcount
            conn.commit()

        # update specialty.n_qa
        cur.execute(
            """
            UPDATE specialty s SET n_qa = sub.cnt
            FROM (SELECT specialty, COUNT(*) AS cnt FROM qa_pair GROUP BY specialty) sub
            WHERE s.name = sub.specialty
            """
        )
        cur.execute(
            """
            UPDATE topic t SET n_qa = sub.cnt
            FROM (SELECT specialty, topic, COUNT(*) AS cnt FROM qa_pair GROUP BY specialty, topic) sub
            WHERE t.specialty = sub.specialty AND t.name = sub.topic
            """
        )
        conn.commit()
    return inserted


def fetch_all_qa(limit: int | None = None) -> list[dict]:
    sql = "SELECT qa_id, chunk_id, specialty, topic, part, seq_in_chunk, question, answer FROM qa_pair"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def fetch_qa_by_ids(qa_ids: list[str]) -> list[dict]:
    if not qa_ids:
        return []
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT qa_id, chunk_id, specialty, topic, question, answer FROM qa_pair WHERE qa_id = ANY(%s)",
            (qa_ids,),
        )
        return cur.fetchall()


def fetch_chunk_siblings(qa_id: str) -> list[dict]:
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT q2.qa_id, q2.question, q2.answer, q2.chunk_id
            FROM qa_pair q1
            JOIN qa_pair q2 ON q2.chunk_id = q1.chunk_id AND q2.qa_id <> q1.qa_id
            WHERE q1.qa_id = %s
            ORDER BY q2.seq_in_chunk
            """,
            (qa_id,),
        )
        return cur.fetchall()
