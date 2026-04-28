"""LLM-driven relation extraction.

Approach:
1. Read entity mentions per qa_id from Postgres (built in P2).
2. For each QA whose entity-count >= 2, send (Q, A, entity list) to the LLM
   and ask for relation triples constrained to the entity list + relation enum.
3. Persist to Postgres `relation` table (with evidence chunkId) and propagate
   to Neo4j as :Entity nodes + typed edges.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Iterable

import orjson
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from ..clients.llm import LLMClient
from ..config import get_settings
from ..graph.client import session as graph_session
from .prompts import (
    RELATION_TYPES,
    build_relation_prompt,
    build_relation_user,
)
from .storage import pg_conn


def _entity_id(name: str, etype: str) -> str:
    h = hashlib.blake2b(f"{etype}::{name}".encode("utf-8"), digest_size=8).hexdigest()
    return f"{etype.lower()}_{h}"


# ---------------------------------------------------------------------------
# Build context bundles per QA
# ---------------------------------------------------------------------------


def fetch_qa_with_entities(min_entities: int = 2, limit: int | None = None) -> list[dict]:
    sql = """
    SELECT q.qa_id, q.chunk_id, q.question, q.answer,
           ARRAY_AGG(DISTINCT e.name)        AS ent_names,
           ARRAY_AGG(DISTINCT e.type)        AS ent_types,
           ARRAY_AGG(DISTINCT e.entity_id)   AS ent_ids
      FROM qa_pair q
      JOIN qa_entity_mention m ON m.qa_id = q.qa_id
      JOIN entity e            ON e.entity_id = m.entity_id
     GROUP BY q.qa_id
    HAVING COUNT(DISTINCT e.entity_id) >= %s
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (min_entities,))
        return cur.fetchall()


def _entity_lookup(rows: list[dict]) -> dict[str, dict]:
    """Return name -> {type, entity_id} for the entity list of *one* qa."""
    out: dict[str, dict] = {}
    for name, etype, eid in zip(rows["ent_names"], rows["ent_types"], rows["ent_ids"]):
        out[name] = {"type": etype, "entity_id": eid}
    return out


# ---------------------------------------------------------------------------
# Stage 1: per-QA relation extraction (LLM)
# ---------------------------------------------------------------------------


async def _extract_one(llm: LLMClient, qa: dict, system_prompt: str, *, model: str | None = None) -> dict:
    entities = [
        {"name": n, "type": t}
        for n, t in zip(qa["ent_names"], qa["ent_types"])
    ]
    user = build_relation_user(qa["question"], qa["answer"], entities)
    try:
        data = await llm.chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=1024,
            model=model,
        )
        rels = data.get("relations") if isinstance(data, dict) else data
        rels = rels or []
        clean = []
        ent_set = set(qa["ent_names"])
        for r in rels:
            if not isinstance(r, dict):
                continue
            head = (r.get("head") or "").strip()
            tail = (r.get("tail") or "").strip()
            rel = (r.get("rel") or "").strip().upper()
            if rel not in RELATION_TYPES:
                continue
            if head not in ent_set or tail not in ent_set or head == tail:
                continue
            clean.append(
                {
                    "head": head,
                    "rel": rel,
                    "tail": tail,
                    "evidence": (r.get("evidence") or "")[:160],
                    "confidence": float(r.get("confidence") or 0.7),
                }
            )
        return {"qa_id": qa["qa_id"], "chunk_id": qa["chunk_id"], "relations": clean}
    except Exception as exc:
        logger.exception("relation extract failed qa_id={}: {}", qa["qa_id"], exc)
        return {"qa_id": qa["qa_id"], "chunk_id": qa["chunk_id"], "relations": [], "error": str(exc)[:200]}


async def extract_relations_async(
    qa_rows: Iterable[dict],
    *,
    cache_path: Path,
    resume: bool = True,
) -> Path:
    qa_rows = list(qa_rows)

    done_ids: set[str] = set()
    if resume and cache_path.exists():
        with cache_path.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    done_ids.add(orjson.loads(line)["qa_id"])
                except Exception:
                    pass
        logger.info("resume: {} qa already extracted", len(done_ids))

    todo = [r for r in qa_rows if r["qa_id"] not in done_ids]
    if not todo:
        logger.info("relations cache complete")
        return cache_path

    s = get_settings()
    models = [s.llm_model_name]
    if s.llm_model_name_2:
        models.append(s.llm_model_name_2)
    logger.info(
        "extracting relations from {} qa pairs (concurrency={} per model, models={})",
        len(todo), s.llm_concurrency, models,
    )
    sys_prompt = build_relation_prompt()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    write_lock = asyncio.Lock()
    clients = [LLMClient(model=m) for m in models]
    try:
        with cache_path.open("ab") as fout:
            async def _runner(idx: int, qa: dict):
                cli = clients[idx % len(clients)]
                res = await _extract_one(cli, qa, sys_prompt, model=cli.model)
                async with write_lock:
                    fout.write(orjson.dumps(res) + b"\n")
                    fout.flush()
                return res

            await tqdm_asyncio.gather(
                *[_runner(i, qa) for i, qa in enumerate(todo)],
                desc="relation-extract",
            )
    finally:
        await asyncio.gather(*[c.aclose() for c in clients])

    return cache_path


# ---------------------------------------------------------------------------
# Stage 2: persist to Postgres + Neo4j
# ---------------------------------------------------------------------------


def _name_to_entity_map() -> dict[str, dict]:
    """Build {name: {entity_id, type}}; for collisions across types, keep the most-frequent type."""
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT entity_id, name, type FROM entity")
        rows = cur.fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        out[r["name"]] = {"entity_id": r["entity_id"], "type": r["type"]}
    return out


def persist_relations_to_pg(cache_path: Path) -> int:
    name_map = _name_to_entity_map()
    n_inserted = 0
    rows: list[tuple] = []
    with cache_path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            obj = orjson.loads(line)
            qa_id = obj["qa_id"]
            chunk_id = obj["chunk_id"]
            for r in obj.get("relations") or []:
                h = name_map.get(r["head"])
                t = name_map.get(r["tail"])
                if not h or not t:
                    continue
                rows.append(
                    (
                        h["entity_id"],
                        r["rel"],
                        t["entity_id"],
                        qa_id,
                        chunk_id,
                        float(r["confidence"]),
                    )
                )
    with pg_conn() as conn, conn.cursor() as cur:
        BATCH = 1000
        for i in range(0, len(rows), BATCH):
            cur.executemany(
                """
                INSERT INTO relation(head_id, rel_type, tail_id, evidence_qa, evidence_chunk, confidence)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (head_id, rel_type, tail_id, evidence_qa) DO NOTHING
                """,
                rows[i : i + BATCH],
            )
            n_inserted += cur.rowcount
        conn.commit()
    logger.info("inserted {} relation rows (after dedup)", n_inserted)
    return n_inserted


def push_graph_layer() -> None:
    """Stream entities and relations from Postgres into Neo4j."""
    with pg_conn() as pgconn, pgconn.cursor() as cur:
        cur.execute("SELECT entity_id, name, type, aliases, description FROM entity")
        ent_rows = cur.fetchall()
        cur.execute(
            """
            SELECT m.qa_id, e.entity_id, e.name, e.type, m.surface
              FROM qa_entity_mention m JOIN entity e ON e.entity_id = m.entity_id
            """
        )
        mention_rows = cur.fetchall()
        cur.execute(
            """
            SELECT head_id, rel_type, tail_id, evidence_qa, evidence_chunk, confidence
              FROM relation
            """
        )
        rel_rows = cur.fetchall()

    BATCH = 1000
    with graph_session() as gs:
        for i in range(0, len(ent_rows), BATCH):
            gs.run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {entity_id: row.entity_id})
                  ON CREATE SET e.name = row.name,
                                e.type = row.type,
                                e.aliases = row.aliases,
                                e.description = row.description
                  ON MATCH  SET e.aliases = row.aliases,
                                e.description = row.description
                WITH e, row
                CALL apoc.create.addLabels(e, [row.type]) YIELD node
                RETURN count(node)
                """,
                rows=ent_rows[i : i + BATCH],
            )
        logger.info("pushed {} entities to neo4j", len(ent_rows))

        for i in range(0, len(mention_rows), BATCH):
            gs.run(
                """
                UNWIND $rows AS row
                MATCH (q:QAPair {qa_id: row.qa_id})
                MATCH (e:Entity {entity_id: row.entity_id})
                MERGE (q)-[m:MENTIONS]->(e)
                  ON CREATE SET m.surface = row.surface
                """,
                rows=mention_rows[i : i + BATCH],
            )
        logger.info("pushed {} mention edges", len(mention_rows))

        # Typed relation edges; we use APOC to materialise typed relationships.
        for i in range(0, len(rel_rows), BATCH):
            gs.run(
                """
                UNWIND $rows AS row
                MATCH (h:Entity {entity_id: row.head_id})
                MATCH (t:Entity {entity_id: row.tail_id})
                CALL apoc.merge.relationship(
                    h, row.rel_type, {evidence_qa: row.evidence_qa},
                    {confidence: row.confidence, evidence_chunk: row.evidence_chunk}, t
                ) YIELD rel
                RETURN count(rel)
                """,
                rows=rel_rows[i : i + BATCH],
            )
        logger.info("pushed {} typed relations", len(rel_rows))
