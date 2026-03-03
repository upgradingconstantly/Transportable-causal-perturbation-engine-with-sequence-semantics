"""Thin wrapper for the Phase 18 cloud execution planner."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> int:
    from tcpe.cli import main as tcpe_main

    return int(tcpe_main(["cloud-exec", *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
