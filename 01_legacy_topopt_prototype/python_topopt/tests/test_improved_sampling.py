from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from constraints.connectivity import historical_precheck
from data_io.matlab import load_seed
from encoding.chromosome import Chromosome
from improved_sampling.config import load_improved_sampling_config
from improved_sampling.database import ImprovedSamplingDatabase
from improved_sampling.features import hamming, topology_features
from improved_sampling.generator import ParentArchiveGenerator
from improved_sampling.operators import mutate_2d
from improved_sampling.repair import repair_topology
from improved_sampling.validation import validate_improved_run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "improved_dataset_sampling.json"


def _seed() -> Chromosome:
    return load_seed(
        Path(r"C:\Users\26096\Desktop\em\FP\CLONALG\seed_bits_fine.mat"),
        variable="seed_bits",
    )


def _small_config(tmp_path: Path) -> Path:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    data.update(
        {
            "parent_archive_target": 4,
            "campaign_count": 2,
            "parents_per_campaign": 2,
            "topology_cluster_target": 2,
            "parents_per_cluster": 2,
        }
    )
    data["proposal"]["selection_window"] = 3
    data["proposal"]["maximum_proposals"] = 2000
    path = tmp_path / "small_config.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def test_approved_configuration_and_hash_split() -> None:
    config = load_improved_sampling_config(CONFIG)
    assert config.parent_target == 60
    assert config.data["campaign_count"] == 6
    assert config.data["parents_per_campaign"] == 10
    assert config.data["topology_cluster_target"] == 12
    assert config.data["parents_per_cluster"] == 5
    assert len({config.config_hash, config.physics_hash, config.sampling_hash}) == 3


@pytest.mark.parametrize("seed_value", [1, 2, 3, 100])
def test_multiscale_mutation_is_reproducible_and_changes_only_design(seed_value: int) -> None:
    config = load_improved_sampling_config(CONFIG)
    first = mutate_2d(_seed(), config.data, np.random.default_rng(seed_value))
    second = mutate_2d(_seed(), config.data, np.random.default_rng(seed_value))
    assert first == second
    assert first.changed_indices
    spec = config.data["proposal"]["scales"][first.scale]
    assert len(first.changed_indices) <= int(spec["maximum_cells"])
    assert first.operator in config.data["proposal"]["operator_weights"]


def test_fixed_cell_is_never_mutated() -> None:
    config = load_improved_sampling_config(CONFIG)
    data = deepcopy(config.data)
    seed = _seed()
    data["constraints"]["fixed_cells"] = {str(index): seed.genes[index] for index in range(180)}
    with pytest.raises(ValueError, match="no mutable"):
        mutate_2d(seed, data, np.random.default_rng(1))


def _invalid_seed_with_small_copper() -> Chromosome:
    seed = _seed()
    for index, value in enumerate(seed.genes):
        if value == 0:
            genes = list(seed.genes)
            genes[index] = 2
            candidate = Chromosome.from_iterable(genes)
            if historical_precheck(candidate).copper.n_floating:
                return candidate
    raise AssertionError("test seed has no isolated-copper mutation")


def test_repair_is_deterministic_idempotent_and_legal() -> None:
    config = load_improved_sampling_config(CONFIG)
    raw = _invalid_seed_with_small_copper()
    first = repair_topology(raw, config.data)
    second = repair_topology(raw, config.data)
    assert first == second
    assert first.status == "repaired"
    assert first.repair_hamming >= 1
    checked = historical_precheck(first.repaired)
    assert not checked.rejected and checked.has_any_copper
    again = repair_topology(first.repaired, config.data)
    assert again.status == "unchanged_legal"
    assert again.repaired == first.repaired
    assert again.repair_hamming == 0


def test_repair_hamming_limit_is_enforced() -> None:
    config = load_improved_sampling_config(CONFIG)
    data = deepcopy(config.data)
    data["repair"]["maximum_repair_hamming"] = 0
    result = repair_topology(_invalid_seed_with_small_copper(), data)
    assert result.status == "repair_hamming_exceeded"
    assert result.rejection_reasons == ("MAXIMUM_REPAIR_HAMMING",)


def _parent_signature(generator: ParentArchiveGenerator) -> list[tuple]:
    return [
        (
            row["archive_order"],
            row["chromosome_hash"],
            row["minimum_archive_hamming"],
            row["parent_parent_id"],
        )
        for row in generator.database.parents()
    ]


def test_parent_archive_is_legal_unique_diverse_and_campaigns_balanced(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    generator = ParentArchiveGenerator.create(
        project_root=tmp_path / "project", run_name="complete", config_path=config
    )
    assert generator.run() == "completed"
    parents = generator.database.parents()
    assert len(parents) == 4
    hashes = {row["chromosome_hash"] for row in parents}
    assert len(hashes) == 4
    for row in parents:
        chromosome = Chromosome.from_iterable(json.loads(row["chromosome_json"]))
        result = historical_precheck(chromosome)
        assert not result.rejected and result.has_any_copper
        assert topology_features(chromosome).copper_cells > 0
    assert all(row["minimum_archive_hamming"] > 0 for row in parents[1:])
    memberships = generator.database.query(
        "SELECT campaign_id,COUNT(*) n FROM campaign_members GROUP BY campaign_id"
    )
    assert [row["n"] for row in memberships] == [2, 2]
    assert generator.database.integrity_check() == "ok"
    validation = validate_improved_run(generator.layout.root)
    assert validation.ok, validation.errors


def test_interrupted_resume_matches_uninterrupted(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    uninterrupted = ParentArchiveGenerator.create(
        project_root=tmp_path / "full", run_name="run", config_path=config
    )
    assert uninterrupted.run() == "completed"

    split = ParentArchiveGenerator.create(
        project_root=tmp_path / "split", run_name="run", config_path=config
    )
    assert split.run(stop_after_parents=2) == "pilot_stop"
    resumed = ParentArchiveGenerator.resume(split.layout.root)
    assert resumed.run() == "completed"
    assert _parent_signature(resumed) == _parent_signature(uninterrupted)
    assert resumed.state["rng_state"] == uninterrupted.state["rng_state"]
    left = uninterrupted.database.query(
        """SELECT parent_id,operator,scale,raw_hash,repaired_hash,repair_status,legal,
                  unique_repaired,nearest_archive_hamming,nearest_all_hamming,selected_as_parent
           FROM proposals ORDER BY proposal_pk"""
    )
    right = resumed.database.query(
        """SELECT parent_id,operator,scale,raw_hash,repaired_hash,repair_status,legal,
                  unique_repaired,nearest_archive_hamming,nearest_all_hamming,selected_as_parent
           FROM proposals ORDER BY proposal_pk"""
    )
    assert [tuple(row) for row in right] == [tuple(row) for row in left]


def test_status_database_can_be_opened_read_only(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    generator = ParentArchiveGenerator.create(
        project_root=tmp_path / "project", run_name="status", config_path=config
    )
    generator.run(stop_after_parents=2)
    before = generator.layout.database.stat().st_mtime_ns
    readonly = ImprovedSamplingDatabase(generator.layout.database, read_only=True)
    assert readonly.summary()["parent_count"] == 2
    assert readonly.integrity_check() == "ok"
    assert generator.layout.database.stat().st_mtime_ns == before
