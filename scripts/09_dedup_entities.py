"""Step 9 (optional): dedup entities by alias overlap & normalised name match.

Run after step 04 (entity extraction). Safe to re-run; idempotent.
Use --dry-run first to see how many merges would happen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.entity_dedup import dedup


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    summary = dedup(dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
