"""Qdrant abstraction for our three vector views.

Collections:
- medgraph_q      : per-QA question embedding
- medgraph_a      : per-QA answer embedding
- medgraph_topic  : per-topic LLM-generated summary embedding
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..config import get_settings


_VEC_NAME = "vec"


@dataclass
class VectorHit:
    point_id: str
    score: float
    payload: dict


def get_client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, prefer_grpc=False, timeout=120)


def existing_point_ids(collection: str) -> set[str]:
    """Scroll a collection and return all point ids as strings."""
    client = get_client()
    ids: set[str] = set()
    next_offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection,
            limit=2000,
            offset=next_offset,
            with_payload=False,
            with_vectors=False,
        )
        for r in records:
            ids.add(str(r.id))
        if next_offset is None:
            break
    return ids


def _ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    try:
        info = client.get_collection(name)
        existing_dim = info.config.params.vectors.size  # type: ignore
        if existing_dim != dim:
            logger.warning(
                "collection {} has dim={} but expected {}; dropping & recreating",
                name,
                existing_dim,
                dim,
            )
            client.delete_collection(name)
        else:
            return
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    # Payload indexes for filtered search
    for field, schema in [
        ("specialty", qm.PayloadSchemaType.KEYWORD),
        ("topic", qm.PayloadSchemaType.KEYWORD),
        ("chunk_id", qm.PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(name, field_name=field, field_schema=schema)
        except Exception:
            pass
    logger.info("collection {} created (dim={})", name, dim)


def ensure_collections(dim: int) -> None:
    s = get_settings()
    client = get_client()
    for c in (s.qdrant_collection_q, s.qdrant_collection_a, s.qdrant_collection_topic):
        _ensure_collection(client, c, dim)


def _qa_point_id(qa_id: str) -> str:
    return str(uuid.UUID(qa_id.ljust(32, "0")[:32]))


def upsert_qa_vectors(
    collection: str,
    qa_rows: list[dict],
    vectors: list[list[float]],
) -> None:
    """qa_rows: dicts with qa_id, chunk_id, specialty, topic, question, answer."""
    if not qa_rows:
        return
    if len(qa_rows) != len(vectors):
        raise ValueError("qa_rows and vectors length mismatch")
    client = get_client()
    points = [
        qm.PointStruct(
            id=_qa_point_id(r["qa_id"]),
            vector=v,
            payload={
                "qa_id": r["qa_id"],
                "chunk_id": r["chunk_id"],
                "specialty": r["specialty"],
                "topic": r["topic"],
                "question": r["question"],
                "answer": r["answer"],
            },
        )
        for r, v in zip(qa_rows, vectors)
    ]
    BATCH = 128
    for i in range(0, len(points), BATCH):
        # naive retry on transient errors
        for attempt in range(3):
            try:
                client.upsert(collection_name=collection, points=points[i : i + BATCH], wait=True)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning("qdrant upsert retry {}: {}", attempt + 1, e)
    logger.info("upserted {} points into {}", len(points), collection)


def upsert_topic_vectors(rows: list[dict], vectors: list[list[float]]) -> None:
    """rows: dicts with specialty, topic, summary."""
    s = get_settings()
    client = get_client()
    points = [
        qm.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{r['specialty']}::{r['topic']}")),
            vector=v,
            payload={
                "specialty": r["specialty"],
                "topic": r["topic"],
                "summary": r["summary"],
            },
        )
        for r, v in zip(rows, vectors)
    ]
    BATCH = 128
    for i in range(0, len(points), BATCH):
        for attempt in range(3):
            try:
                client.upsert(collection_name=s.qdrant_collection_topic,
                              points=points[i : i + BATCH], wait=True)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning("topic upsert retry {}: {}", attempt + 1, e)
    logger.info("upserted {} topic points", len(points))


def vector_search(
    collection: str,
    query_vec: list[float],
    *,
    top_k: int = 30,
    filter_specialty: str | None = None,
    filter_topic: str | None = None,
) -> list[VectorHit]:
    client = get_client()
    must = []
    if filter_specialty:
        must.append(qm.FieldCondition(key="specialty", match=qm.MatchValue(value=filter_specialty)))
    if filter_topic:
        must.append(qm.FieldCondition(key="topic", match=qm.MatchValue(value=filter_topic)))
    qfilter = qm.Filter(must=must) if must else None
    # qdrant-client >= 1.16 deprecated search() in favour of query_points()
    res = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
    ).points
    return [VectorHit(point_id=str(r.id), score=float(r.score), payload=r.payload or {}) for r in res]
