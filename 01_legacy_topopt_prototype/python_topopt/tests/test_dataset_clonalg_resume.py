from __future__ import annotations

from pathlib import Path

from clonalg.config import ClonalgConfig
from clonalg.engine import ClonalgEngine
from dataset_generation.backend import DeterministicAngleMockBackend
from dataset_generation.clonalg_runner import ClonalgDatasetRunner
from dataset_generation.orchestrator import DatasetRunManager
from evaluator.mock import DeterministicMockEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "dataset_merged_v5.json"


def _ready_manager(tmp_path: Path, name: str) -> tuple[DatasetRunManager, DeterministicAngleMockBackend]:
    manager = DatasetRunManager.create(
        project_root=tmp_path,
        run_name=name,
        config_path=CONFIG,
    )
    backend = DeterministicAngleMockBackend()
    reference = manager.create_reference_sample()
    assert manager.evaluate_sample(reference, backend=backend, chromosome=None) == "completed"
    return manager, backend


def _deterministic_candidate_rows(manager: DatasetRunManager) -> list[tuple]:
    rows = manager.database.query_all(
        """SELECT candidate_id,generation,candidate_order,chromosome_hash,source,
                  parent_candidate_id,parent_rank,clone_index,mutation_operator,
                  mutation_strength,hamming_distance_to_parent,lineage_id,status,
                  validity_status,duplicate_of_sample_id,torque_ratio,historical_j,fitness_band
           FROM candidates ORDER BY candidate_pk"""
    )
    return [tuple(row) for row in rows]


def test_dataset_runner_matches_existing_historical_engine_for_two_generations(
    tmp_path: Path,
) -> None:
    manager, backend = _ready_manager(tmp_path, "compat")
    runner = ClonalgDatasetRunner(manager=manager, backend=backend, generation_limit=2)
    assert runner.run() == "completed"

    expected = ClonalgEngine(
        seed_chromosome=manager.seed_chromosome,
        evaluator=DeterministicMockEvaluator(),
        config=ClonalgConfig(generations=2, random_seed=manager.config.random_seed),
    ).run()
    assert len(runner.state["history"]) == len(expected.history) == 2
    for actual, historical in zip(runner.state["history"], expected.history):
        assert actual["population"] == [list(row) for row in historical.population]
        assert actual["population_historical_j"] == list(historical.objectives)
        assert actual["sorted_indices"] == list(historical.sorted_indices)
        assert actual["clone_counts"] == list(historical.clone_counts)
        assert actual["clone_historical_j"] == list(historical.clone_objectives)
        assert actual["immigrant_positions"] == list(historical.immigrant_positions)
        assert actual["immigrant_modes"] == list(historical.immigrant_modes)


def test_multiple_interruptions_match_uninterrupted_candidate_order_rng_and_population(
    tmp_path: Path,
) -> None:
    continuous_manager, continuous_backend = _ready_manager(tmp_path, "continuous")
    continuous = ClonalgDatasetRunner(
        manager=continuous_manager,
        backend=continuous_backend,
        generation_limit=3,
    )
    assert continuous.run() == "completed"

    resumed_manager, resumed_backend = _ready_manager(tmp_path, "resumed")
    partial = ClonalgDatasetRunner(
        manager=resumed_manager,
        backend=resumed_backend,
        generation_limit=3,
    )
    for threshold in (7, 19, 33):
        def stop_at_threshold(limit: int = threshold) -> bool:
            return len(
                resumed_manager.database.query_all(
                    "SELECT candidate_id FROM candidates WHERE run_id=?",
                    (resumed_manager.run_id,),
                )
            ) >= limit

        assert partial.run(stop_requested=stop_at_threshold) == "paused"
        resumed_manager = DatasetRunManager.resume(resumed_manager.layout.root)
        partial = ClonalgDatasetRunner.resume(
            manager=resumed_manager,
            backend=resumed_backend,
        )
    assert partial.run() == "completed"

    assert partial.state["history"] == continuous.state["history"]
    assert partial.state["population"] == continuous.state["population"]
    assert partial.state["rng_state"] == continuous.state["rng_state"]
    assert _deterministic_candidate_rows(resumed_manager) == _deterministic_candidate_rows(
        continuous_manager
    )


def test_two_consecutive_resumes_do_not_duplicate_candidates(tmp_path: Path) -> None:
    manager, backend = _ready_manager(tmp_path, "double_resume")
    runner = ClonalgDatasetRunner(manager=manager, backend=backend, generation_limit=2)

    assert runner.run(stop_requested=lambda: len(manager.database.query_all("SELECT 1 FROM candidates")) >= 5) == "paused"
    count = len(manager.database.query_all("SELECT 1 FROM candidates"))
    first = DatasetRunManager.resume(manager.layout.root)
    second = DatasetRunManager.resume(manager.layout.root)
    assert len(second.database.query_all("SELECT 1 FROM candidates")) == count
    resumed = ClonalgDatasetRunner.resume(manager=second, backend=backend)
    assert resumed.run() == "completed"
