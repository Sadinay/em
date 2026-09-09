from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_generation.reporting import read_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only dataset run status")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(read_status(args.run), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
