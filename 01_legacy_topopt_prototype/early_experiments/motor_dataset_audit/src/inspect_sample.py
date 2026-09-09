from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one audited FEM–MAT sample.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()
    path = args.output / "dataset_manifest.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    row = next((item for item in records if item.get("sample_id") == args.sample_id), None)
    if row is None:
        raise SystemExit(f"Sample not found: {args.sample_id}")
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
