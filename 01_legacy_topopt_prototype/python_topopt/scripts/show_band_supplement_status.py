from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Show band supplement progress")
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    path = args.run.resolve() / "band_supplement_status.json"
    if not path.is_file():
        raise SystemExit(f"status file not found: {path}")
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
