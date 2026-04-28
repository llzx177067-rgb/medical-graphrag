"""Neo4j constraints, indexes and schema bootstrap."""
from __future__ import annotations

from loguru import logger

from .client import session


CONSTRAINTS = [
    "CREATE CONSTRAINT specialty_name IF NOT EXISTS FOR (s:Specialty) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT topic_key IF NOT EXISTS FOR (t:Topic) REQUIRE t.key IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT qa_id IF NOT EXISTS FOR (q:QAPair) REQUIRE q.qa_id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX qa_chunk IF NOT EXISTS FOR (q:QAPair) ON (q.chunk_id)",
]


def init_schema() -> None:
    with session() as s:
        for c in CONSTRAINTS:
            s.run(c)
        for i in INDEXES:
            s.run(i)
    logger.info("neo4j constraints & indexes ensured")
