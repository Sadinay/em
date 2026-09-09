from __future__ import annotations

from dataclasses import replace

from clonalg.cache import EvaluationCache
from clonalg.config import ClonalgConfig
from clonalg.engine import ClonalgEngine
from encoding.chromosome import Chromosome
from evaluator.mock import DeterministicMockEvaluator


def test_cache_hits_and_batch_dedup(valid_copper_chromosome) -> None:
    evaluator = DeterministicMockEvaluator()
    cache = EvaluationCache()
    first = cache.evaluate_batch(
        [valid_copper_chromosome, valid_copper_chromosome], evaluator
    )
    assert first.stats.evaluator_calls == 1
    assert first.stats.batch_duplicates == 1
    assert evaluator.calls == 1
    second = cache.evaluate_batch([valid_copper_chromosome], evaluator)
    assert second.stats.persistent_hits == 1
    assert second.stats.evaluator_calls == 0
    assert evaluator.calls == 1


def test_same_python_seed_repeats_entire_run(valid_copper_chromosome) -> None:
    config = ClonalgConfig(generations=5, random_seed=8675309)
    first = ClonalgEngine(
        seed_chromosome=valid_copper_chromosome,
        evaluator=DeterministicMockEvaluator(),
        config=config,
    ).run()
    second = ClonalgEngine(
        seed_chromosome=valid_copper_chromosome,
        evaluator=DeterministicMockEvaluator(),
        config=config,
    ).run()
    assert first == second


def test_checkpoint_resume_matches_uninterrupted_run(
    valid_copper_chromosome, tmp_path
) -> None:
    config = ClonalgConfig(generations=7, random_seed=424242)
    uninterrupted = ClonalgEngine(
        seed_chromosome=valid_copper_chromosome,
        evaluator=DeterministicMockEvaluator(),
        config=config,
    ).run()

    checkpoint = tmp_path / "clonalg-checkpoint.json"
    interrupted_engine = ClonalgEngine(
        seed_chromosome=valid_copper_chromosome,
        evaluator=DeterministicMockEvaluator(),
        config=config,
    )
    partial = interrupted_engine.run(
        stop_after_generation=3, checkpoint_path=checkpoint
    )
    assert not partial.completed
    assert partial.next_generation == 4
    assert checkpoint.exists()

    resumed_engine = ClonalgEngine.from_checkpoint(
        checkpoint, evaluator=DeterministicMockEvaluator()
    )
    resumed = resumed_engine.run(checkpoint_path=checkpoint)
    assert resumed == uninterrupted

