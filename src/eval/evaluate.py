"""Evaluate retrieval Recall@k / MRR over the hold-out test set.

Runs items concurrently via asyncio.Semaphore (configurable).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from ..config import get_settings
from ..retrieval.pipeline import GraphRAGRetriever


async def _eval_one(retriever: GraphRAGRetriever, sem: asyncio.Semaphore, it: dict, top_k: int) -> dict:
    async with sem:
        try:
            bundle = await retriever.retrieve(it["question"], rerank_top_k=top_k, with_siblings=False)
        except Exception as exc:
            return {"err": str(exc)[:200], "qa_id": it["qa_id"]}
    ids = [x.qa_id for x in bundle.items]
    chunks = [x.chunk_id for x in bundle.items]
    res = {
        "qa_id": it["qa_id"],
        "qa_hit": int(it["qa_id"] in ids),
        "chunk_hit": int(it["chunk_id"] in chunks),
    }
    if it["chunk_id"] in chunks:
        res["chunk_rank"] = chunks.index(it["chunk_id"]) + 1
    return res


async def run(testset_path: Path, top_k: int, parallelism: int) -> None:
    items = [json.loads(line) for line in testset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info("evaluating on {} items, top_k={}, parallelism={}", len(items), top_k, parallelism)

    sem = asyncio.Semaphore(parallelism)
    async with GraphRAGRetriever() as retriever:
        results = await tqdm_asyncio.gather(
            *[_eval_one(retriever, sem, it, top_k) for it in items], desc="evaluating"
        )

    n = len(results)
    errs = [r for r in results if "err" in r]
    ok = [r for r in results if "err" not in r]
    qa_hits = sum(r["qa_hit"] for r in ok)
    chunk_hits = sum(r["chunk_hit"] for r in ok)
    rr_chunk = sum(1.0 / r["chunk_rank"] for r in ok if r.get("chunk_rank"))

    summary = {
        "n": n,
        "n_errors": len(errs),
        "qa_recall@k": round(qa_hits / max(n, 1), 4),
        "chunk_recall@k": round(chunk_hits / max(n, 1), 4),
        "chunk_mrr@k": round(rr_chunk / max(n, 1), 4),
        "top_k": top_k,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # Also write a detailed file
    out_path = testset_path.parent / "eval_results.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("detail written to {}", out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--testset", default=None)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--parallelism", type=int, default=16,
                   help="number of items evaluated concurrently")
    args = p.parse_args()

    s = get_settings()
    path = Path(args.testset or (s.cache_path / "testset.jsonl"))
    asyncio.run(run(path, args.top_k, args.parallelism))


if __name__ == "__main__":
    main()
