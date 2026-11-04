"""Run the REGAL RSMD stage through the shared command-line entry point."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_regal import main as run_regal


def main(argv: list[str] | None = None) -> None:
    run_regal(argv, fixed_stage="rsmd")


if __name__ == "__main__":
    main()
