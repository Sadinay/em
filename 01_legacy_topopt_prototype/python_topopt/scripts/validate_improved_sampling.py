from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from improved_sampling.validation import validate_improved_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate improved parent sampling run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--skip-repair-replay", action="store_true")
    args = parser.parse_args()
    report = validate_improved_run(args.run, verify_repairs=not args.skip_repair_replay)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
