from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motor_regression.config import load_config
from motor_regression.training import prepare_data, run_training


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train motor topology PyTorch regressors")
    result.add_argument(
        "command", choices=("audit", "train"), help="audit data only, or audit then train"
    )
    result.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "motor_regression_ripple.yaml"
    )
    result.add_argument("--output", type=Path, help="override output directory")
    result.add_argument(
        "--models",
        nargs="+",
        choices=("resnet20", "mlp", "small_cnn"),
        default=("resnet20", "mlp", "small_cnn"),
    )
    return result


def main() -> int:
    args = parser().parse_args()
    config = load_config(args.config)
    configured = Path(config["output"]["directory"])
    output = args.output.resolve() if args.output else (ROOT / configured).resolve()
    if args.command == "audit":
        _, split, _, audit = prepare_data(config, ROOT, output)
        print(json.dumps({"audit": audit, "split_counts": {k: len(v) for k, v in split.items()}},
                         ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    result = run_training(
        config,
        project_root=ROOT,
        output_directory=output,
        model_names=args.models,
    )
    summary = {
        model: values["splits"]["test"] for model, values in result["models"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
