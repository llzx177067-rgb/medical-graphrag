"""Build a stratified hold-out test set from Postgres.

Sampling strategy: stratify by (specialty), pick N per specialty (defaults to
20), with a fixed RNG seed. The resulting JSONL is the gold for retrieval
recall@k evaluation: for each question, the gold chunk_id is the chunk it came from.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from loguru import logger

from ..config import get_settings
from ..ingest.storage import fetch_all_qa


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-specialty", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = fetch_all_qa()
    by_sp: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sp[r["specialty"]].append(r)

    test: list[dict] = []
    for sp, lst in by_sp.items():
        rng.shuffle(lst)
        test.extend(lst[: args.per_specialty])

    s = get_settings()
    out = Path(args.out or (s.cache_path / "testset.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in test:
            f.write(
                json.dumps(
                    {
                        "qa_id": r["qa_id"],
                        "question": r["question"],
                        "answer": r["answer"],
                        "chunk_id": r["chunk_id"],
                        "specialty": r["specialty"],
                        "topic": r["topic"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("wrote {} test items to {}", len(test), out)


if __name__ == "__main__":
    main()
