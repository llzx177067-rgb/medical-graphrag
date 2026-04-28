"""Neo4j driver factory + helper context."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from loguru import logger
from neo4j import GraphDatabase, Session

from ..config import get_settings


_driver = None


def get_driver():
    global _driver
    if _driver is None:
        s = get_settings()
        _driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
        logger.info("neo4j driver initialised at {}", s.neo4j_uri)
    return _driver


@contextmanager
def session() -> Iterator[Session]:
    drv = get_driver()
    with drv.session() as s:
        yield s
