"""Parse Professional.json and break chunkId into a 3-level hierarchy.

chunkId format observed in the dataset:
    "<Specialty> - <Topic>-part-<N>"
e.g.
    "临床药理学 - 药物-受体相互作用-part-1"
    "传染病 - 疟疾-part-3"

Note: the *Topic* portion may itself contain hyphens (e.g. "药物-受体相互作用"),
so we split *only* on the suffix "-part-<int>" anchored at the end.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import orjson


_PART_RX = re.compile(r"-part-(\d+)\s*$")


@dataclass(slots=True, frozen=True)
class QARecord:
    qa_id: str           # deterministic hash id
    question: str
    answer: str
    chunk_id: str        # raw chunkId
    specialty: str       # 第一段
    topic: str           # 第二段（去除 -part-N）
    part: int            # part 序号
    seq_in_chunk: int    # 在同一 chunk_id 中的顺序


def _stable_id(*parts: str) -> str:
    h = hashlib.blake2b("||".join(parts).encode("utf-8"), digest_size=8).hexdigest()
    return h


def parse_chunk_id(chunk_id: str) -> tuple[str, str, int]:
    """Return (specialty, topic, part)."""
    if " - " not in chunk_id:
        return chunk_id.strip(), "未分类", 1
    head, tail = chunk_id.split(" - ", 1)
    head = head.strip()
    tail = tail.strip()
    m = _PART_RX.search(tail)
    if m:
        part = int(m.group(1))
        topic = tail[: m.start()].rstrip("-").strip()
    else:
        part = 1
        topic = tail
    if not topic:
        topic = "未分类"
    return head, topic, part


def iter_records(json_path: Path | str) -> Iterator[QARecord]:
    """Stream records from Professional.json."""
    raw = Path(json_path).read_bytes()
    data = orjson.loads(raw)
    seq_counter: dict[str, int] = {}

    for row in data:
        cid = (row.get("chunkId") or "").strip()
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not q or not a or not cid:
            continue
        specialty, topic, part = parse_chunk_id(cid)
        seq = seq_counter.get(cid, 0)
        seq_counter[cid] = seq + 1
        qa_id = _stable_id(cid, q, str(seq))

        yield QARecord(
            qa_id=qa_id,
            question=q,
            answer=a,
            chunk_id=cid,
            specialty=specialty,
            topic=topic,
            part=part,
            seq_in_chunk=seq,
        )


def collect_taxonomy(records: Iterable[QARecord]) -> tuple[set[str], dict[str, set[str]], dict[str, dict]]:
    """Return (specialties, specialty -> topics, chunk_id -> chunk meta)."""
    specialties: set[str] = set()
    sp_to_topics: dict[str, set[str]] = {}
    chunks: dict[str, dict] = {}
    for r in records:
        specialties.add(r.specialty)
        sp_to_topics.setdefault(r.specialty, set()).add(r.topic)
        meta = chunks.setdefault(
            r.chunk_id,
            {
                "chunk_id": r.chunk_id,
                "specialty": r.specialty,
                "topic": r.topic,
                "part": r.part,
                "n_qa": 0,
            },
        )
        meta["n_qa"] += 1
    return specialties, sp_to_topics, chunks
