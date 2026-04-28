"""Step 8: evaluate retrieval over the hold-out test set."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.evaluate import main

if __name__ == "__main__":
    main()
