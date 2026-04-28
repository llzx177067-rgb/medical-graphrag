"""Smoke test: verify connectivity to DashScope (LLM / embedding / rerank)
and the three local services (Postgres / Neo4j / Qdrant).

Prints model-call timings to console; full structured logs go to
`data/cache/model_calls.log`.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.clients.embedding import EmbeddingClient
from src.clients.llm import LLMClient
from src.clients.rerank import RerankClient
from src.config import get_settings


async def _probe_llm() -> None:
    async with LLMClient() as llm:
        t0 = time.perf_counter()
        out = await llm.chat(
            messages=[
                {"role": "system", "content": "你是医学助手。请用一句话回答。"},
                {"role": "user", "content": "什么是高血压？"},
            ],
            temperature=0.1,
            max_tokens=120,
        )
        dt = (time.perf_counter() - t0) * 1000
        print(f"[LLM] {dt:.0f} ms ->", (out or "")[:120].replace("\n", " "))


async def _probe_embed() -> None:
    async with EmbeddingClient() as emb:
        t0 = time.perf_counter()
        vecs = await emb.embed(["糖尿病的并发症有哪些？", "梅毒的诊断方法"])
        dt = (time.perf_counter() - t0) * 1000
        print(f"[EMBED] {dt:.0f} ms -> n={len(vecs)} dim={len(vecs[0])}")


async def _probe_rerank() -> None:
    async with RerankClient() as rk:
        t0 = time.perf_counter()
        res = await rk.rerank(
            "糖尿病并发症",
            [
                "糖尿病常见并发症包括视网膜病变、肾病、神经病变。",
                "高血压常分为原发性和继发性两类。",
                "1 型糖尿病是胰岛 β 细胞破坏导致胰岛素绝对不足。",
            ],
            top_n=3,
        )
        dt = (time.perf_counter() - t0) * 1000
        print(f"[RERANK] {dt:.0f} ms -> {res}")


def _probe_pg():
    try:
        from src.ingest.storage import pg_conn
        with pg_conn() as c, c.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            row = cur.fetchone()
            print("[PG] ok=", row)
    except Exception as e:
        print("[PG] FAILED:", e)


def _probe_neo4j():
    try:
        from src.graph.client import session
        with session() as s:
            res = s.run("RETURN 1 AS ok").single()
            print("[NEO4J] ok=", res["ok"])
    except Exception as e:
        print("[NEO4J] FAILED:", e)


def _probe_qdrant():
    try:
        from src.index.vector_store import get_client
        c = get_client()
        info = c.get_collections()
        print("[QDRANT] collections=", [c.name for c in info.collections])
    except Exception as e:
        print("[QDRANT] FAILED:", e)


async def main() -> None:
    s = get_settings()
    print("== config ==")
    print(f"  base_url        = {s.base_url}")
    print(f"  llm_model       = {s.llm_model_name}")
    print(f"  embedding_model = {s.embedding_model}")
    print(f"  rerank_model    = {s.rerank_model}")
    print(f"  api_key         = {s.api_key[:6]}***{s.api_key[-4:] if s.api_key else ''}")
    print()

    print("== infra ==")
    _probe_pg()
    _probe_neo4j()
    _probe_qdrant()
    print()

    print("== model probes ==  (full log: data/cache/model_calls.log)")
    await _probe_llm()
    await _probe_embed()
    await _probe_rerank()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    # Re-import to keep model-call sink (file) registered
    from src.clients import _base  # noqa
    asyncio.run(main())
