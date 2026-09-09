from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset_generation.backend import DeterministicAngleMockBackend
from dataset_generation.config import load_dataset_config
from dataset_generation.export import export_dataset
from dataset_generation.femm_backend import IsolatedFemmAngleBackend
from dataset_generation.orchestrator import DatasetRunManager
from dataset_generation.reporting import read_status
from dataset_generation.validation import validate_run
from femm_runner.process import WorkerOutcome


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "dataset_merged_v5.json"


def _complete_mock_run(tmp_path: Path) -> DatasetRunManager:
    manager = DatasetRunManager.create(
        project_root=tmp_path,
        run_name="run",
        config_path=CONFIG,
    )
    backend = DeterministicAngleMockBackend()
    reference = manager.create_reference_sample()
    assert manager.evaluate_sample(reference, backend=backend, chromosome=None) == "completed"
    result = manager.evaluate_candidate(
        manager.seed_chromosome,
        backend=backend,
        generation=1,
        candidate_order=1,
        source="seed",
        lineage_id="L-SEED",
    )
    assert result.status == "completed"
    manager.write_state()
    return manager


def test_validation_export_and_historical_j(tmp_path: Path) -> None:
    manager = _complete_mock_run(tmp_path)
    report = validate_run(manager.layout.root)
    assert report.ok, report.errors
    manifest = export_dataset(manager.layout.root, tmp_path / "export")
    assert manifest["sample_count"] == 1
    row = json.loads((tmp_path / "export" / "samples.jsonl").read_text(encoding="utf-8"))
    assert row["historical_J"] == pytest.approx(
        manager.database.get_sample(row["sample_id"])["historical_j"]
    )
    assert np.asarray(row["material_matrix_18x10"]).shape == (18, 10)
    assert np.asarray(row["material_matrix_10x18"]).shape == (10, 18)
    arrays = np.load(tmp_path / "export" / "dataset.npz")
    assert arrays["chromosome_180"].shape == (1, 180)
    assert arrays["torque_values_nm"].shape == (1, 6)
    assert arrays["historical_J"].shape == (1,)


def test_status_is_read_only_and_eta_handles_data(tmp_path: Path) -> None:
    manager = _complete_mock_run(tmp_path)
    before = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in manager.layout.root.iterdir()
        if path.is_file()
    }
    status = read_status(manager.layout.root)
    after = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in manager.layout.root.iterdir()
        if path.is_file()
    }
    assert before == after
    assert status["recent_mean_seconds_per_angle"] == pytest.approx(0.061)
    assert status["solver_only_seconds_for_total_target"] is not None
    assert status["band_completion_eta"] is None


def test_validation_detects_recomputed_scalar_mismatch(tmp_path: Path) -> None:
    manager = _complete_mock_run(tmp_path)
    sample = manager.database.query_all(
        "SELECT sample_id FROM samples WHERE sample_kind='candidate'"
    )[0]["sample_id"]
    with manager.database.transaction() as connection:
        connection.execute("UPDATE samples SET historical_j=NULL WHERE sample_id=?", (sample,))
    report = validate_run(manager.layout.root)
    assert not report.ok
    assert any("historical J" in error for error in report.errors)


def test_femm_backend_uses_reference_and_candidate_circuit_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs: list[dict] = []

    def fake_worker(job_path: Path, **kwargs: object) -> WorkerOutcome:
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        jobs.append(job)
        result_path = Path(job["result_path"])
        if job["operation"] == "prepare_candidate":
            payload = {"status": "SUCCEEDED", "timings": {"topology_seconds": 1.25}}
        else:
            payload = {
                "status": "SUCCEEDED",
                "torque_values_nm": [0.8123],
                "timings": {
                    "angles": [
                        {
                            "mesh_seconds": 2.0,
                            "solve_seconds": 3.0,
                            "postprocess_seconds": 0.1,
                            "total_angle_seconds": 5.1,
                            "mesh_nodes": 123,
                            "mesh_elements": 234,
                        }
                    ]
                },
            }
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return WorkerOutcome(456, 0, False, "", "", ())

    monkeypatch.setattr("dataset_generation.femm_backend.run_isolated_worker", fake_worker)
    backend = IsolatedFemmAngleBackend(load_dataset_config(CONFIG))
    reference = backend.evaluate_angle(
        sample_id="SREF",
        sample_kind="reference",
        chromosome=None,
        angle_deg=0.0,
        work_directory=tmp_path / "reference" / "angle_0",
    )
    candidate = backend.evaluate_angle(
        sample_id="SCAND",
        sample_kind="candidate",
        chromosome=DatasetRunManager.create(
            project_root=tmp_path / "project",
            run_name="seed",
            config_path=CONFIG,
        ).seed_chromosome,
        angle_deg=3.0,
        work_directory=tmp_path / "candidate" / "angle_3",
    )
    assert reference.torque_nm == pytest.approx(0.8123)
    assert candidate.build_time == pytest.approx(1.25)
    assert [job["circuit_mode"] for job in jobs] == [
        "three_phase_abc",
        "split_six_phase_signs",
        "split_six_phase_signs",
    ]
    assert jobs[0]["model_path"].lower().endswith("strukturfemm.fem") is False
    assert jobs[1]["operation"] == "prepare_candidate"
    assert jobs[2]["torque_angles_deg"] == [3.0]
