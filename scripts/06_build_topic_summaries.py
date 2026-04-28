"""Step 6: LLM-generate per-topic summaries and embed into Qdrant medgraph_topic."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.summary import build_topic_summaries


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=None,
                   help="override LLM concurrency for this lighter job")
    args = p.parse_args()
    asyncio.run(build_topic_summaries(limit=args.limit, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
