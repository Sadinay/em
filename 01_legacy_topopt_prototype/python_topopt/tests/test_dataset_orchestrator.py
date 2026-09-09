from __future__ import annotations

from pathlib import Path
import threading
import time

import numpy as np
import pytest

from dataset_generation.backend import DeterministicAngleMockBackend
from dataset_generation.orchestrator import DatasetRunManager
from encoding.layout import grid_to_matlab_vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "dataset_merged_v5.json"


def _manager(tmp_path: Path, name: str = "run") -> DatasetRunManager:
    return DatasetRunManager.create(
        project_root=tmp_path,
        run_name=name,
        config_path=CONFIG,
    )


def test_new_run_reference_candidate_and_duplicate_cache(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    backend = DeterministicAngleMockBackend()
    reference_id = manager.create_reference_sample()
    assert manager.evaluate_sample(reference_id, backend=backend, chromosome=None) == "completed"
    assert manager.database.get_run(manager.run_id)["t_avg_ref"] == pytest.approx(1.0)

    first = manager.evaluate_candidate(
        manager.seed_chromosome,
        backend=backend,
        generation=1,
        candidate_order=1,
        source="seed",
    )
    calls_after_first = len(backend.calls)
    second = manager.evaluate_candidate(
        manager.seed_chromosome,
        backend=backend,
        generation=1,
        candidate_order=2,
        source="clone",
        parent_candidate_id=first.candidate_id,
        parent_rank=1,
        clone_index=1,
        mutation_operator="trit_hypermutation",
        mutation_strength=0.03,
        mutation_indices=(),
        hamming_distance_to_parent=0,
    )
    assert first.status == "completed"
    assert first.torque_ratio is not None
    assert first.historical_j is not None
    assert second.duplicate
    assert second.sample_id == first.sample_id
    assert second.torque_ratio == first.torque_ratio
    assert second.historical_j == first.historical_j
    assert len(backend.calls) == calls_after_first


def test_angle_level_pause_and_resume_skips_committed_angles(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    backend = DeterministicAngleMockBackend()
    reference_id = manager.create_reference_sample()
    stop = lambda: len(backend.calls) >= 3
    assert (
        manager.evaluate_sample(
            reference_id,
            backend=backend,
            chromosome=None,
            stop_requested=stop,
        )
        == "paused"
    )
    assert [call[1] for call in backend.calls] == [0.0, 3.0, 6.0]
    resumed = DatasetRunManager.resume(manager.layout.root)
    resumed_backend = DeterministicAngleMockBackend()
    assert (
        resumed.evaluate_sample(reference_id, backend=resumed_backend, chromosome=None)
        == "completed"
    )
    assert [call[1] for call in resumed_backend.calls] == [9.0, 12.0, 15.0]


def test_running_angle_is_interrupted_on_resume_and_retried(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    sample_id = manager.create_reference_sample()
    manager.database.mark_angle_running(
        run_id=manager.run_id,
        sample_id=sample_id,
        angle_deg=0.0,
    )
    resumed = DatasetRunManager.resume(manager.layout.root)
    assert resumed.database.angle_rows(sample_id)[0]["status"] == "interrupted"
    backend = DeterministicAngleMockBackend()
    assert resumed.evaluate_sample(sample_id, backend=backend, chromosome=None) == "completed"
    assert backend.calls[0] == (sample_id, 0.0)


def test_timeout_retries_without_zero_fitness(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    sample_id = manager.create_reference_sample()
    backend = DeterministicAngleMockBackend(timeout_once_at={(sample_id, 9.0)})
    assert manager.evaluate_sample(sample_id, backend=backend, chromosome=None) == "completed"
    attempts = {float(row["angle_deg"]): int(row["attempt"]) for row in manager.database.angle_rows(sample_id)}
    assert attempts[9.0] == 2
    assert manager.database.get_sample(sample_id)["t_avg"] == pytest.approx(1.0)


def test_rejected_candidate_is_recorded_without_backend_call(tmp_path: Path) -> None:
    from encoding.chromosome import Chromosome

    manager = _manager(tmp_path)
    backend = DeterministicAngleMockBackend()
    candidate = manager.evaluate_candidate(
        Chromosome.from_iterable([0] * 180),
        backend=backend,
        generation=1,
        candidate_order=1,
        source="random_immigrant",
    )
    assert candidate.status == "rejected"
    assert candidate.historical_j == 1_000_000.0
    assert candidate.torque_ratio is None
    assert backend.calls == []


class _ConcurrentTrackingBackend:
    max_workers = 3

    def __init__(self) -> None:
        self.inner = DeterministicAngleMockBackend()
        self.lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def evaluate_angle(self, **kwargs):
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(0.2)
            return self.inner.evaluate_angle(**kwargs)
        finally:
            with self.lock:
                self.active -= 1

    def close(self) -> None:
        return None


def test_three_candidates_run_concurrently_but_commit_in_main_thread(tmp_path: Path) -> None:
    manager = _manager(tmp_path, "parallel_candidates")
    reference_backend = DeterministicAngleMockBackend()
    reference_id = manager.create_reference_sample()
    assert manager.evaluate_sample(reference_id, backend=reference_backend, chromosome=None) == "completed"

    requests = []
    for order, row0 in enumerate((2, 6, 10), start=1):
        grid = np.ones((18, 10), dtype=np.uint8)
        grid[row0 : row0 + 2, 4:6] = 2
        chromosome = grid_to_matlab_vector(grid)
        candidate_id, sample_id, reasons, duplicate = manager.register_candidate(
            chromosome,
            generation=1,
            candidate_order=order,
            source="parallel_test",
        )
        assert candidate_id and sample_id and not reasons and not duplicate
        requests.append((sample_id, chromosome))

    backend = _ConcurrentTrackingBackend()
    statuses = manager.evaluate_samples_parallel(
        requests,
        backend=backend,
        max_workers=3,
    )
    assert set(statuses.values()) == {"completed"}
    assert backend.maximum_active == 3
    assert all(manager.database.get_sample(sample_id)["status"] == "classified" for sample_id, _ in requests)
