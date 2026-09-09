from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from src.build_manifest import build_manifest
from src.common import read_csv, setup_logging
from src.generate_report import generate_report
from src.inspect_fem import inspect_fem_files
from src.inspect_mat import inspect_mat_files
from src.match_files import match_files
from src.scan_files import scan_dataset


STAGES = ("scan", "inspect", "match", "manifest", "report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only FEMM/MATLAB motor dataset audit.")
    parser.add_argument("--input", type=Path, required=True, help="Original dataset directory (never modified).")
    parser.add_argument("--output", type=Path, required=True, help="Separate output directory.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--from-stage", choices=STAGES, default="scan", help="Resume from this stage.")
    parser.add_argument("--force", action="store_true", help="Recompute stage outputs even when cached CSVs exist.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise SystemExit("Output must be outside the original input directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, args.verbose)
    config = load_config(args.config)
    shutil.copy2(args.config, output_dir / "audit_config_used.yaml")
    start_index = STAGES.index(args.from_stage)

    inventory_path = output_dir / "file_inventory.csv"
    if start_index <= 0 or not inventory_path.exists() or args.force:
        inventory = scan_dataset(input_dir, output_dir, config, logger)
    else:
        inventory = read_csv(inventory_path)
        logger.info("Using cached file inventory (%d rows)", len(inventory))

    mat_summary_path = output_dir / "mat_file_summary.csv"
    fem_summary_path = output_dir / "fem_file_summary.csv"
    variables_path = output_dir / "mat_variable_inventory.csv"
    if start_index <= 1 or not all(p.exists() for p in (mat_summary_path, fem_summary_path, variables_path)) or args.force:
        mat_summaries, mat_variables = inspect_mat_files(input_dir, output_dir, inventory, config, logger)
        fem_summaries = inspect_fem_files(input_dir, output_dir, inventory, config, logger)
    else:
        mat_summaries = read_csv(mat_summary_path)
        mat_variables = read_csv(variables_path)
        fem_summaries = read_csv(fem_summary_path)
        logger.info("Using cached MAT/FEM inspection outputs")

    pairs_path = output_dir / "fem_mat_pairs.csv"
    if start_index <= 2 or not pairs_path.exists() or args.force:
        pairs = match_files(inventory, output_dir, config, logger)
    else:
        pairs = read_csv(pairs_path)
        logger.info("Using cached pair outputs")

    manifest_path = output_dir / "dataset_manifest.csv"
    if start_index <= 3 or not manifest_path.exists() or args.force:
        manifest = build_manifest(output_dir, inventory, fem_summaries, mat_variables, pairs, logger)
    else:
        manifest = read_csv(manifest_path)
        logger.info("Using cached manifest")

    generate_report(input_dir, output_dir, inventory, fem_summaries, mat_summaries, mat_variables, pairs, manifest, logger)
    logger.info("Audit complete: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
