from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_generation.validation import validate_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only dataset run validation")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = validate_run(args.run)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
