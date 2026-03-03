"""Thin wrapper for the Phase 17 artifact sync planner."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> int:
    from tcpe.cloud_handoff import artifact_sync_main

    return artifact_sync_main()


if __name__ == "__main__":
    raise SystemExit(main())
