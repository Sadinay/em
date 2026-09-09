from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from dataset_generation.bands import classify_torque_ratio
from dataset_generation.checkpoint import read_checkpoint, write_checkpoint
from dataset_generation.config import load_dataset_config
from dataset_generation.database import DatasetDatabase
from dataset_generation.hashing import json_sha256
from dataset_generation.state import RunStatus, atomic_write_json, read_json_with_backup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "dataset_merged_v5.json"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "B1"),
        (0.499999, "B1"),
        (0.5, "B2"),
        (0.6, "B3"),
        (0.7, "B4"),
        (0.8, "B5"),
        (0.9, "B6"),
        (1.0, "B7"),
        (2.0, "B7"),
        (-0.001, "negative_torque"),
    ],
)
def test_fitness_band_boundaries(value: float, expected: str) -> None:
    assert classify_torque_ratio(value) == expected


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_fitness_band_rejects_nonfinite(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        classify_torque_ratio(value)


def test_config_hash_is_stable_and_changes_with_physics() -> None:
    config = load_dataset_config(CONFIG)
    assert config.config_hash == json_sha256(config.data)
    duplicate = deepcopy(config.data)
    assert json_sha256(duplicate) == config.config_hash
    duplicate["physics"]["stator_current_peak_a"] = 3.6
    assert json_sha256(duplicate) != config.config_hash


def test_atomic_json_ignores_damaged_temporary_and_uses_backup(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"generation": 1}, keep_backup=True)
    atomic_write_json(path, {"generation": 2}, keep_backup=True)
    (tmp_path / ".state.json.damaged.tmp").write_text("{", encoding="utf-8")
    assert read_json_with_backup(path) == {"generation": 2}
    path.write_text("{", encoding="utf-8")
    assert read_json_with_backup(path) == {"generation": 1}


def test_checkpoint_hash_validation_and_backup(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    write_checkpoint(path, {"generation": 1, "population": [[0, 1, 2]]})
    write_checkpoint(path, {"generation": 2, "population": [[2, 1, 0]]})
    assert read_checkpoint(path)["generation"] == 2
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["generation"] = 99
    path.write_text(json.dumps(envelope), encoding="utf-8")
    # The verified previous checkpoint is used instead of the corrupt primary.
    assert read_checkpoint(path)["generation"] == 1


def _new_database(tmp_path: Path) -> tuple[DatasetDatabase, str, dict]:
    config = load_dataset_config(CONFIG)
    database = DatasetDatabase(tmp_path / "dataset.sqlite")
    database.initialize()
    run_id = "test_run"
    database.create_run(
        run_id=run_id,
        config=config.data,
        config_hash=config.config_hash,
        source_hash="a" * 64,
        code_version="source-manifest:a",
    )
    return database, run_id, config.data


def test_sqlite_pragmas_unique_angle_and_run_status(tmp_path: Path) -> None:
    database, run_id, config = _new_database(tmp_path)
    sample_id = database.create_sample(
        run_id=run_id,
        candidate_id=None,
        sample_kind="reference",
        physical_key_hash="reference-key",
        chromosome=None,
        material_matrix=None,
        chromosome_hash=None,
        config_hash="c" * 64,
        geometry_mode=config["reference_model"]["geometry_mode"],
        angles_deg=(0.0, 3.0),
        femm_version="4.2",
    )
    assert sample_id == "S000001"
    with pytest.raises(Exception):
        with database.transaction() as connection:
            connection.execute(
                "INSERT INTO angle_evaluations(sample_id,angle_deg,status) VALUES(?,?,?)",
                (sample_id, 0.0, "pending"),
            )
    database.set_run_status(run_id, RunStatus.RUNNING)
    with pytest.raises(ValueError, match="illegal"):
        database.set_run_status(run_id, RunStatus.CREATED)
    assert database.integrity_check() == "ok"


def test_each_angle_is_committed_and_completed_angle_cannot_restart(tmp_path: Path) -> None:
    database, run_id, config = _new_database(tmp_path)
    sample_id = database.create_sample(
        run_id=run_id,
        candidate_id=None,
        sample_kind="reference",
        physical_key_hash="reference-key",
        chromosome=None,
        material_matrix=None,
        chromosome_hash=None,
        config_hash="c" * 64,
        geometry_mode=config["reference_model"]["geometry_mode"],
        angles_deg=(0.0, 3.0),
        femm_version="4.2",
    )
    assert database.mark_angle_running(run_id=run_id, sample_id=sample_id, angle_deg=0.0) == 1
    database.complete_angle(
        run_id=run_id,
        sample_id=sample_id,
        angle_deg=0.0,
        torque=0.8,
        metrics={"total_time": 2.0, "solve_time": 1.5},
    )
    rows = database.angle_rows(sample_id)
    assert [(row["angle_deg"], row["status"]) for row in rows] == [
        (0.0, "completed"),
        (3.0, "pending"),
    ]
    with pytest.raises(ValueError, match="completed"):
        database.mark_angle_running(run_id=run_id, sample_id=sample_id, angle_deg=0.0)
    with pytest.raises(ValueError, match="every configured angle"):
        database.finalize_sample(
            run_id=run_id,
            sample_id=sample_id,
            t_avg_ref=None,
            historical_j=None,
        )


def test_sample_summary_ratio_band_and_historical_j(tmp_path: Path) -> None:
    database, run_id, config = _new_database(tmp_path)
    sample_id = database.create_sample(
        run_id=run_id,
        candidate_id=None,
        sample_kind="candidate",
        physical_key_hash="sample-key",
        chromosome=[0, 1, 2] * 60,
        material_matrix=[[0] * 10 for _ in range(18)],
        chromosome_hash="d" * 64,
        config_hash="c" * 64,
        geometry_mode=config["candidate_model"]["geometry_mode"],
        angles_deg=(0.0, 3.0, 6.0, 9.0, 12.0, 15.0),
        femm_version="4.2",
    )
    values = (0.8, 0.9, 1.0, 1.1, 1.0, 0.9)
    for angle, torque in zip((0, 3, 6, 9, 12, 15), values):
        database.mark_angle_running(run_id=run_id, sample_id=sample_id, angle_deg=angle)
        database.complete_angle(
            run_id=run_id,
            sample_id=sample_id,
            angle_deg=angle,
            torque=torque,
            metrics={"total_time": 1.0, "solve_time": 0.8},
        )
    result = database.finalize_sample(
        run_id=run_id,
        sample_id=sample_id,
        t_avg_ref=1.0,
        historical_j=0.25,
    )
    assert result["T_avg"] == pytest.approx(0.95)
    assert result["torque_ratio"] == pytest.approx(0.95)
    assert result["fitness_band"] == "B6"
    assert result["historical_J"] == 0.25
