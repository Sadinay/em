from __future__ import annotations

import json
from pathlib import Path

from dataset_generation.backend import DeterministicAngleMockBackend
from dataset_generation.hashing import json_sha256
from dataset_generation.orchestrator import DatasetRunManager
from dataset_generation.validation import validate_run
from improved_sampling.campaign_runner import ImprovedCampaignRunner
from improved_sampling.generator import ParentArchiveGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHYSICS_CONFIG = PROJECT_ROOT / "configs" / "dataset_merged_v5.json"


def _parent_run(tmp_path: Path) -> Path:
    data = json.loads(
        (PROJECT_ROOT / "configs" / "improved_dataset_sampling.json").read_text(encoding="utf-8")
    )
    data.update(
        parent_archive_target=4,
        campaign_count=2,
        parents_per_campaign=2,
        topology_cluster_target=2,
        parents_per_cluster=2,
    )
    data["proposal"]["selection_window"] = 4
    data["physics"]["source_config_path"] = str(PHYSICS_CONFIG)
    physics = json.loads(PHYSICS_CONFIG.read_text(encoding="utf-8"))
    data["physics"]["source_config_sha256"] = json_sha256(physics)
    config = tmp_path / "parent_config.json"
    config.write_text(json.dumps(data), encoding="utf-8")
    generator = ParentArchiveGenerator.create(
        project_root=tmp_path, run_name="parents", config_path=config
    )
    assert generator.run() == "completed"
    return generator.layout.root


def _campaign_config(tmp_path: Path) -> Path:
    data = json.loads(
        (PROJECT_ROOT / "configs" / "improved_campaign_evolution.json").read_text(encoding="utf-8")
    )
    data["offspring_per_campaign_generation"] = 2
    data["elite_count"] = 1
    data["hamming_admission"] = {
        scale: {"minimum_archive_distance": 1, "maximum_parent_distance": 180}
        for scale in ("small", "medium", "large")
    }
    path = tmp_path / "campaign_config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_campaign_runner_uses_parent_archive_and_resumes(tmp_path: Path) -> None:
    parent_run = _parent_run(tmp_path)
    manager = DatasetRunManager.create(
        project_root=tmp_path, run_name="dataset", config_path=PHYSICS_CONFIG
    )
    backend = DeterministicAngleMockBackend(max_workers=3)
    reference = manager.create_reference_sample()
    assert manager.evaluate_sample(reference, backend=backend, chromosome=None) == "completed"
    runner = ImprovedCampaignRunner.create(
        manager=manager,
        backend=backend,
        parent_run=parent_run,
        config_path=_campaign_config(tmp_path),
    )

    assert runner.run(stop_after_samples=5) == "pilot_stop"
    assert manager.database.progress_summary(manager.run_id)["valid_unique_samples"] == 5
    first_count = len(manager.database.query_all("SELECT 1 FROM candidates"))

    resumed_manager = DatasetRunManager.resume(manager.layout.root)
    resumed = ImprovedCampaignRunner.resume(
        manager=resumed_manager, backend=DeterministicAngleMockBackend(max_workers=3)
    )
    assert resumed.run(stop_after_samples=7) == "pilot_stop"
    summary = resumed_manager.database.progress_summary(resumed_manager.run_id)
    assert summary["valid_unique_samples"] == 7
    assert len(resumed_manager.database.query_all("SELECT 1 FROM candidates")) > first_count
    proposals = resumed_manager.database.query_all(
        "SELECT * FROM improved_proposals ORDER BY proposal_order"
    )
    assert proposals
    assert all(row["raw_hash"] and row["repaired_hash"] for row in proposals)
    assert resumed.state["rng_state"]
    assert resumed.state["execution_workers"] == 3
    assert all(
        row["best_historical_j"] <= row["previous_best_historical_j"]
        for row in resumed.state["history"]
    )
    report = validate_run(resumed_manager.layout.root)
    assert report.ok, report.as_dict()


class _FailOneChromosomeBackend(DeterministicAngleMockBackend):
    def __init__(self, failed_hash: str) -> None:
        super().__init__()
        self.failed_hash = failed_hash

    def evaluate_angle(self, **kwargs):
        chromosome = kwargs.get("chromosome")
        if chromosome is not None and chromosome.sha256() == self.failed_hash:
            raise RuntimeError("Material properties have not been defined for all regions")
        return super().evaluate_angle(**kwargs)


def test_physical_failure_is_archived_and_campaign_continues(tmp_path: Path) -> None:
    parent_run = _parent_run(tmp_path)
    manager = DatasetRunManager.create(
        project_root=tmp_path, run_name="failure_isolation", config_path=PHYSICS_CONFIG
    )
    reference_backend = DeterministicAngleMockBackend()
    reference = manager.create_reference_sample()
    assert manager.evaluate_sample(reference, backend=reference_backend, chromosome=None) == "completed"
    runner = ImprovedCampaignRunner.create(
        manager=manager,
        backend=reference_backend,
        parent_run=parent_run,
        config_path=_campaign_config(tmp_path),
    )
    failed_hash = runner.state["campaigns"][0]["population"][0]["chromosome"]
    from encoding.chromosome import Chromosome

    runner.backend = _FailOneChromosomeBackend(
        Chromosome.from_iterable(failed_hash).sha256()
    )
    assert runner.run(stop_after_samples=5) == "pilot_stop"
    summary = manager.database.progress_summary(manager.run_id)
    assert summary["valid_unique_samples"] == 5
    failed = manager.database.query_all(
        "SELECT candidate_id,status FROM candidates WHERE status='failed'"
    )
    assert len(failed) == 1
    assert manager.database.query_all(
        "SELECT sample_id FROM samples WHERE status='femm_failed'"
    )
