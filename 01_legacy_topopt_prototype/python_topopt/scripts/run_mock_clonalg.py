from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clonalg.config import ClonalgConfig  # noqa: E402
from clonalg.engine import ClonalgEngine  # noqa: E402
from data_io.matlab import load_best_result  # noqa: E402
from evaluator.mock import DeterministicMockEvaluator  # noqa: E402


DEFAULT_SEED_FILE = (
    WORKSPACE_ROOT
    / "FP"
    / "IA_test_Float"
    / "best_result_20260306_174158.mat"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pure-Python CLONALG with a deterministic mock evaluator."
    )
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260806)
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "mock_run_result.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = DeterministicMockEvaluator()
    if args.resume:
        if args.checkpoint is None:
            raise SystemExit("--resume requires --checkpoint")
        engine = ClonalgEngine.from_checkpoint(args.checkpoint, evaluator=evaluator)
    else:
        best = load_best_result(args.seed_file)
        config = ClonalgConfig(
            generations=args.generations,
            random_seed=args.random_seed,
        )
        engine = ClonalgEngine(
            seed_chromosome=best.chromosome,
            evaluator=evaluator,
            config=config,
        )

    result = engine.run(checkpoint_path=args.checkpoint)
    payload = {
        "evaluator": "DeterministicMockEvaluator (no FEMM)",
        "completed": result.completed,
        "generations_completed": len(result.history),
        "next_generation": result.next_generation,
        "best_objective": result.best_objective,
        "best_chromosome_cache_key": result.best_chromosome.matlab_cache_key(),
        "best_chromosome_sha256": result.best_chromosome.sha256(),
        "cache_size": result.cache_size,
        "evaluator_calls_this_process": evaluator.calls,
        "history": [record.to_dict() for record in result.history],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"completed={result.completed} generations={len(result.history)} "
        f"best_J={result.best_objective:.12g} cache={result.cache_size}"
    )


if __name__ == "__main__":
    main()
