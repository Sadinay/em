from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from improved_sampling.database import ImprovedSamplingDatabase


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only improved sampling status")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    database = ImprovedSamplingDatabase(args.run / "sampling.sqlite", read_only=True)
    result = database.summary()
    result["sqlite_integrity"] = database.integrity_check()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
