"""Graph-based recall: from anchor entities, traverse Neo4j to fetch
candidate QAPair ids whose chunks are evidence for relevant relations.

Ranking heuristic at this layer (before unified reranker):
- weight 1.0 for direct mention match in QAPair
- weight 0.7 for evidence_qa of a 1-hop relation involving an anchor entity
- weight 0.4 for evidence_qa of a 2-hop path
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from loguru import logger

from ..graph.client import session


@dataclass
class GraphHit:
    qa_id: str
    score: float
    chunk_id: str
    reason: str


_CYPHER_DIRECT = """
MATCH (e:Entity)
WHERE e.name IN $names OR ANY(a IN e.aliases WHERE a IN $names)
WITH collect(DISTINCT e) AS anchors
UNWIND anchors AS a
MATCH (q:QAPair)-[:MENTIONS]->(a)
RETURN q.qa_id AS qa_id, q.chunk_id AS chunk_id, a.name AS via, 1.0 AS score
LIMIT $limit
"""

_CYPHER_ONE_HOP = """
MATCH (e:Entity)
WHERE e.name IN $names OR ANY(a IN e.aliases WHERE a IN $names)
WITH collect(DISTINCT e) AS anchors
UNWIND anchors AS a
MATCH (a)-[r]->(b:Entity)
WHERE NOT type(r) IN ['MENTIONS','BELONGS_TO']
WITH a, b, type(r) AS rel, r.evidence_qa AS qa_id, r.evidence_chunk AS chunk_id
WHERE qa_id IS NOT NULL
RETURN qa_id, chunk_id, a.name + '-[' + rel + ']->' + b.name AS via, 0.7 AS score
LIMIT $limit
"""

_CYPHER_BACKWARD = """
MATCH (e:Entity)
WHERE e.name IN $names OR ANY(a IN e.aliases WHERE a IN $names)
WITH collect(DISTINCT e) AS anchors
UNWIND anchors AS a
MATCH (b:Entity)-[r]->(a)
WHERE NOT type(r) IN ['MENTIONS','BELONGS_TO']
WITH a, b, type(r) AS rel, r.evidence_qa AS qa_id, r.evidence_chunk AS chunk_id
WHERE qa_id IS NOT NULL
RETURN qa_id, chunk_id, b.name + '-[' + rel + ']->' + a.name AS via, 0.7 AS score
LIMIT $limit
"""

_CYPHER_TWO_HOP = """
MATCH (e:Entity)
WHERE e.name IN $names OR ANY(a IN e.aliases WHERE a IN $names)
WITH collect(DISTINCT e) AS anchors
UNWIND anchors AS a
MATCH (a)-[r1]->(b:Entity)-[r2]->(c:Entity)
WHERE NOT type(r1) IN ['MENTIONS','BELONGS_TO']
  AND NOT type(r2) IN ['MENTIONS','BELONGS_TO']
WITH a, b, c, type(r1) AS rel1, type(r2) AS rel2,
     coalesce(r2.evidence_qa, r1.evidence_qa) AS qa_id,
     coalesce(r2.evidence_chunk, r1.evidence_chunk) AS chunk_id
WHERE qa_id IS NOT NULL
RETURN qa_id, chunk_id,
       a.name + '-[' + rel1 + ']->' + b.name + '-[' + rel2 + ']->' + c.name AS via,
       0.4 AS score
LIMIT $limit
"""


def graph_recall(entity_names: list[str], top_k: int = 20) -> list[GraphHit]:
    if not entity_names:
        return []
    accum: dict[str, GraphHit] = {}
    queries = [
        (_CYPHER_DIRECT, top_k * 2),
        (_CYPHER_ONE_HOP, top_k * 2),
        (_CYPHER_BACKWARD, top_k * 2),
        (_CYPHER_TWO_HOP, top_k),
    ]
    with session() as s:
        for cypher, lim in queries:
            try:
                res = s.run(cypher, names=entity_names, limit=int(lim))
                for rec in res:
                    qa_id = rec["qa_id"]
                    if not qa_id:
                        continue
                    sc = float(rec["score"])
                    if qa_id not in accum or accum[qa_id].score < sc:
                        accum[qa_id] = GraphHit(
                            qa_id=qa_id,
                            score=sc,
                            chunk_id=rec["chunk_id"],
                            reason=rec["via"],
                        )
            except Exception as exc:
                logger.warning("graph_recall cypher failed: {}", exc)

    sorted_hits = sorted(accum.values(), key=lambda h: -h.score)[:top_k]
    return sorted_hits


def fetch_subgraph_triples(entity_names: list[str], limit: int = 30) -> list[dict]:
    """For context: return human-readable triples around the anchors."""
    if not entity_names:
        return []
    cypher = """
    MATCH (e:Entity)
    WHERE e.name IN $names OR ANY(a IN e.aliases WHERE a IN $names)
    WITH collect(DISTINCT e) AS anchors
    UNWIND anchors AS a
    MATCH (a)-[r]->(b:Entity)
    WHERE NOT type(r) IN ['MENTIONS','BELONGS_TO']
    RETURN a.name AS head, type(r) AS rel, b.name AS tail,
           r.evidence_chunk AS chunk_id, r.confidence AS confidence
    LIMIT $limit
    """
    with session() as s:
        res = s.run(cypher, names=entity_names, limit=int(limit))
        return [dict(r) for r in res]
