"""Step 7: build hold-out test set from Postgres."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.build_test_set import main

if __name__ == "__main__":
    main()
