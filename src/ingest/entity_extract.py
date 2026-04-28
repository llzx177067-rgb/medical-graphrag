"""LLM-driven entity extraction from QA pairs.

Strategy:
- One QA per LLM call (answers are short, ~270 chars on average).
- Concurrency comes from the LLMClient semaphore + asyncio.gather.
- Disk-cached: every result is appended to a JSONL file so we can resume.
- Normalisation runs as a second pass that merges aliases case-insensitively.
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
from .prompts import ENTITY_TYPES, build_entity_prompt, build_entity_user


def _entity_id(name: str, etype: str) -> str:
    h = hashlib.blake2b(f"{etype}::{name}".encode("utf-8"), digest_size=8).hexdigest()
    return f"{etype.lower()}_{h}"


def _norm_name(name: str) -> str:
    return name.strip().replace(" ", "").lower()


# ---------------------------------------------------------------------------
# Stage 1: per-QA extraction
# ---------------------------------------------------------------------------


async def _extract_one(llm: LLMClient, qa: dict, system_prompt: str, *, model: str | None = None) -> dict:
    user = build_entity_user(qa["question"], qa["answer"])
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
        ents = []
        if isinstance(data, dict):
            ents = data.get("entities") or []
        elif isinstance(data, list):
            ents = data
        # sanitize
        clean = []
        for e in ents:
            if not isinstance(e, dict):
                continue
            name = (e.get("name") or "").strip()
            etype = (e.get("type") or "").strip()
            if not name or etype not in ENTITY_TYPES:
                continue
            aliases = e.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
            clean.append(
                {
                    "name": name,
                    "type": etype,
                    "aliases": list({*aliases}),
                    "description": (e.get("description") or "").strip()[:80],
                }
            )
        return {"qa_id": qa["qa_id"], "entities": clean}
    except Exception as exc:
        logger.exception("entity extract failed for qa_id={}: {}", qa["qa_id"], exc)
        return {"qa_id": qa["qa_id"], "entities": [], "error": str(exc)[:200]}


async def extract_entities_async(
    qa_rows: Iterable[dict],
    *,
    cache_path: Path,
    resume: bool = True,
) -> Path:
    """Run LLM entity extraction over all qa_rows, streaming to JSONL cache."""
    s = get_settings()
    qa_rows = list(qa_rows)

    done_ids: set[str] = set()
    if resume and cache_path.exists():
        with cache_path.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = orjson.loads(line)
                    done_ids.add(obj.get("qa_id"))
                except Exception:
                    continue
        logger.info("resume: {} qa already extracted, skipping", len(done_ids))

    todo = [r for r in qa_rows if r["qa_id"] not in done_ids]
    if not todo:
        logger.info("nothing to do; cache already complete")
        return cache_path

    # Build the model pool. If LLM_MODEL_NAME_2 is set we round-robin between
    # two clients so each gets its own concurrency semaphore + rate-limit pool
    # (DashScope rate limits are per-model).
    models = [s.llm_model_name]
    if s.llm_model_name_2:
        models.append(s.llm_model_name_2)
    logger.info(
        "extracting entities from {} qa pairs (concurrency={} per model, models={})",
        len(todo), s.llm_concurrency, models,
    )
    sys_prompt = build_entity_prompt()

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
                desc="entity-extract",
            )
    finally:
        await asyncio.gather(*[c.aclose() for c in clients])

    return cache_path


# ---------------------------------------------------------------------------
# Stage 2: normalise + persist to Postgres
# ---------------------------------------------------------------------------


def normalise_and_persist(cache_path: Path) -> tuple[int, int]:
    """Read JSONL cache, merge aliases, write entity + qa_entity_mention to Postgres."""
    from .storage import pg_conn

    # name+type -> canonical record
    canon: dict[tuple[str, str], dict] = {}
    mentions: list[tuple[str, str, str]] = []  # (qa_id, entity_id, surface)

    with cache_path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            obj = orjson.loads(line)
            qa_id = obj["qa_id"]
            for e in obj.get("entities") or []:
                key = (_norm_name(e["name"]), e["type"])
                rec = canon.get(key)
                if rec is None:
                    rec = {
                        "name": e["name"],
                        "type": e["type"],
                        "aliases": set(e.get("aliases") or []),
                        "description": e.get("description") or "",
                    }
                    canon[key] = rec
                else:
                    # keep shortest non-empty description
                    if e.get("description") and (
                        not rec["description"] or len(e["description"]) < len(rec["description"])
                    ):
                        rec["description"] = e["description"]
                rec["aliases"].update(e.get("aliases") or [])
                rec["aliases"].add(e["name"])
                ent_id = _entity_id(rec["name"], rec["type"])
                mentions.append((qa_id, ent_id, e["name"]))
                rec["entity_id"] = ent_id

    logger.info("normalised: {} unique entities, {} mentions", len(canon), len(mentions))

    rows_entity = [
        (
            rec["entity_id"],
            rec["name"],
            rec["type"],
            sorted(rec["aliases"]),
            rec.get("description", ""),
        )
        for rec in canon.values()
    ]
    with pg_conn() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO entity(entity_id, name, type, aliases, description)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (entity_id) DO UPDATE
                SET aliases    = EXCLUDED.aliases,
                    description = EXCLUDED.description
            """,
            rows_entity,
        )
        # Deduplicate mentions client-side; PG has uniqueness on (qa_id, entity_id, surface)
        unique_mentions = list({(q, e, s) for (q, e, s) in mentions})
        cur.executemany(
            """
            INSERT INTO qa_entity_mention(qa_id, entity_id, surface)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            unique_mentions,
        )
        conn.commit()
    return len(rows_entity), len(mentions)
