from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Callable

import numpy as np

from dataset_generation.backend import AngleEvaluationBackend
from dataset_generation.checkpoint import read_checkpoint
from dataset_generation.hashing import file_sha256, json_sha256
from dataset_generation.orchestrator import DatasetRunManager
from dataset_generation.state import atomic_write_json
from encoding.chromosome import Chromosome

from .band_supplement_config import (
    BandSupplementConfig,
    TARGET_BANDS,
    band_supplement_config_from_data,
    load_band_supplement_config,
)
from .campaign_runner import ImprovedCampaignRunner, _chromosome, _objective
from .features import hamming


def _physics_identity(config: dict[str, Any]) -> str:
    keys = (
        "training_label",
        "seed",
        "reference_model",
        "candidate_model",
        "physics",
        "constraints",
        "historical_objective",
    )
    return json_sha256({key: config[key] for key in keys})


def _ratio(member: dict[str, Any]) -> float:
    value = (member.get("result") or {}).get("torque_ratio")
    return float(value) if value is not None and math.isfinite(float(value)) else -math.inf


def _band(member: dict[str, Any]) -> str | None:
    value = (member.get("result") or {}).get("fitness_band")
    return str(value) if value is not None else None


def _band_number(value: str | None) -> int:
    return int(value[1:]) if value and len(value) == 2 and value.startswith("B") else -99


class BandSupplementRunner(ImprovedCampaignRunner):
    """Band-directed mutation sampler backed by a completed physics archive."""

    STATE_SCHEMA_VERSION = 1
    KIND = "band_supplement_v1"
    METADATA_FILE = "band_supplement_manifest.json"

    def __init__(
        self,
        *,
        manager: DatasetRunManager,
        backend: AngleEvaluationBackend,
        config: BandSupplementConfig,
        mutation_config: dict[str, Any],
        state: dict[str, Any],
        rng: np.random.Generator,
        source_chromosomes: list[Chromosome],
    ) -> None:
        super().__init__(
            manager=manager,
            backend=backend,
            config=config,  # type: ignore[arg-type]
            mutation_config=mutation_config,
            state=state,
            rng=rng,
        )
        self.source_chromosomes = source_chromosomes

    @staticmethod
    def _source_rows(source_run: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
        database = Path(source_run) / "dataset.sqlite"
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            run = connection.execute("SELECT status FROM runs").fetchone()
            if run is None or run["status"] != "completed":
                raise ValueError("source dataset run must be completed")
            rows = [
                dict(row)
                for row in connection.execute(
                    """SELECT s.sample_id,s.first_candidate_id,s.chromosome_json,
                              s.chromosome_hash,s.torque_ratio,s.historical_j,s.fitness_band,
                              COALESCE(c.lineage_id,'BASE') lineage_id
                       FROM samples s LEFT JOIN candidates c
                         ON c.candidate_id=s.first_candidate_id
                       WHERE s.sample_kind='candidate' AND s.status='classified'
                         AND s.chromosome_json IS NOT NULL
                       ORDER BY s.sample_pk"""
                )
            ]
        counts = {f"B{index}": 0 for index in range(1, 8)}
        for row in rows:
            band = row["fitness_band"]
            if band in counts:
                counts[band] += 1
        if not rows:
            raise ValueError("source dataset has no completed candidate samples")
        return rows, counts

    @staticmethod
    def _parent_sort_key(row: dict[str, Any], target_band: str) -> tuple[Any, ...]:
        ratio = float(row["torque_ratio"])
        band = str(row["fitness_band"])
        target_number = _band_number(target_band)
        band_distance = abs(_band_number(band) - target_number)
        if target_band == "B7":
            return (0 if band == "B7" else 1, -ratio, float(row["historical_j"]), row["chromosome_hash"])
        center = 0.05 + 0.1 * (target_number - 1)
        return (
            0 if band == target_band else 1,
            band_distance,
            abs(ratio - center),
            float(row["historical_j"]),
            row["chromosome_hash"],
        )

    @classmethod
    def _select_diverse_parents(
        cls,
        rows: list[dict[str, Any]],
        *,
        target_band: str,
        count: int,
    ) -> list[dict[str, Any]]:
        ordered = sorted(rows, key=lambda row: cls._parent_sort_key(row, target_band))
        if len(ordered) < count:
            raise ValueError(f"only {len(ordered)} eligible parents for {target_band}; need {count}")
        exact = [row for row in ordered if row["fitness_band"] == target_band]
        candidate_rows = exact if len(exact) >= count else ordered
        # Restrict diversity selection to the best deterministic window so
        # physically relevant parents are not displaced by remote weak ones.
        window = candidate_rows[: max(count, min(len(candidate_rows), count * 5))]
        if len(exact) < count:
            selected = list(exact)
            selected_ids = {id(row) for row in selected}
            window = [row for row in window if id(row) not in selected_ids]
            if not selected:
                selected = [window.pop(0)]
        else:
            selected = [window.pop(0)]
        while len(selected) < count:
            chosen = max(
                window,
                key=lambda row: (
                    min(
                        hamming(
                            Chromosome.from_iterable(json.loads(row["chromosome_json"])),
                            Chromosome.from_iterable(json.loads(other["chromosome_json"])),
                        )
                        for other in selected
                    ),
                    tuple(-float(value) if isinstance(value, (int, float)) else str(value)
                          for value in cls._parent_sort_key(row, target_band)),
                ),
            )
            selected.append(chosen)
            window.remove(chosen)
        return selected

    @classmethod
    def create(
        cls,
        *,
        manager: DatasetRunManager,
        backend: AngleEvaluationBackend,
        source_run: Path,
        config_path: Path,
    ) -> "BandSupplementRunner":
        config = load_band_supplement_config(config_path)
        if config.target != manager.config.target_valid_samples:
            raise ValueError("supplement target and immutable physics run target differ")
        source_root = Path(source_run).resolve()
        source_envelope = json.loads((source_root / "run_config.json").read_text(encoding="utf-8"))
        if _physics_identity(source_envelope["config"]) != _physics_identity(manager.config.data):
            raise ValueError("source and supplement runs must use identical physics configuration")
        rows, base_counts = cls._source_rows(source_root)

        source_manifest = json.loads(
            (source_root / ImprovedCampaignRunner.METADATA_FILE).read_text(encoding="utf-8")
        )
        parent_root = Path(source_manifest["parent_run"])
        parent_envelope = json.loads((parent_root / "run_config.json").read_text(encoding="utf-8"))
        mutation_config = parent_envelope["config"]
        population_size = int(config.data["population_per_band"])
        b7_threshold = float(config.data.get("b7_parent_minimum_torque_ratio", 0.98))
        campaigns: list[dict[str, Any]] = []
        for target_band in TARGET_BANDS:
            eligible = set(config.data["eligible_parent_bands"][target_band])
            candidates = [row for row in rows if row["fitness_band"] in eligible]
            if target_band == "B7":
                candidates = [
                    row for row in candidates
                    if row["fitness_band"] == "B7" or float(row["torque_ratio"]) >= b7_threshold
                ]
            parents = cls._select_diverse_parents(
                candidates, target_band=target_band, count=population_size
            )
            population = []
            for rank, row in enumerate(parents, start=1):
                population.append(
                    {
                        "chromosome": json.loads(row["chromosome_json"]),
                        "source": "band_supplement_external_parent",
                        "parent_candidate_id": None,
                        "external_candidate_id": row["first_candidate_id"],
                        "external_sample_id": row["sample_id"],
                        "parent_rank": rank,
                        "clone_index": None,
                        "mutation_operator": None,
                        "mutation_strength": None,
                        "mutation_indices": [],
                        "hamming_distance_to_parent": 0,
                        "lineage_id": f"BASE-{row['first_candidate_id']}",
                        "candidate_id": None,
                        "sample_id": None,
                        "result": {
                            "status": "completed",
                            "torque_ratio": float(row["torque_ratio"]),
                            "historical_j": float(row["historical_j"]),
                            "fitness_band": str(row["fitness_band"]),
                            "duplicate": False,
                            "rejection_reasons": [],
                        },
                    }
                )
            campaigns.append(
                {
                    "campaign_id": f"TARGET-{target_band}",
                    "target_band": target_band,
                    "population": population,
                    "offspring": [],
                }
            )
        rng = np.random.default_rng(config.random_seed)
        state: dict[str, Any] = {
            "schema_version": cls.STATE_SCHEMA_VERSION,
            "runner_kind": cls.KIND,
            "generation": 1,
            "generation_completed": False,
            "phase": "offspring_generation",
            "campaign_index": 0,
            "member_index": 0,
            "candidate_order": 1,
            "proposal_count": 0,
            "pending_offspring": None,
            "pending_offspring_batch": [],
            "campaigns": campaigns,
            "history": [],
            "base_band_counts": base_counts,
            "completed": False,
            "stop_reason": None,
        }
        manifest = {
            "schema_version": 1,
            "runner_kind": cls.KIND,
            "supplement_config_hash": config.config_hash,
            "supplement_config": config.data,
            "source_run": str(source_root),
            "source_database_sha256": file_sha256(source_root / "dataset.sqlite"),
            "source_config_hash": source_envelope["config_hash"],
            "physics_identity_hash": _physics_identity(manager.config.data),
            "dataset_config_hash": manager.config.config_hash,
            "mutation_config": mutation_config,
            "mutation_config_hash": json_sha256(mutation_config),
            "base_band_counts": base_counts,
        }
        atomic_write_json(manager.layout.root / cls.METADATA_FILE, manifest)
        runner = cls(
            manager=manager,
            backend=backend,
            config=config,
            mutation_config=mutation_config,
            state=state,
            rng=rng,
            source_chromosomes=[
                Chromosome.from_iterable(json.loads(row["chromosome_json"])) for row in rows
            ],
        )
        runner.save_checkpoint()
        runner.write_campaign_status()
        return runner

    @classmethod
    def resume(
        cls, *, manager: DatasetRunManager, backend: AngleEvaluationBackend
    ) -> "BandSupplementRunner":
        manifest_path = manager.layout.root / cls.METADATA_FILE
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("runner_kind") != cls.KIND:
            raise ValueError("run is not a band supplement run")
        if manifest.get("dataset_config_hash") != manager.config.config_hash:
            raise ValueError("supplement manifest and dataset config differ")
        if manifest.get("physics_identity_hash") != _physics_identity(manager.config.data):
            raise ValueError("supplement physics identity changed")
        config = band_supplement_config_from_data(manifest["supplement_config"], manifest_path)
        if config.config_hash != manifest.get("supplement_config_hash"):
            raise ValueError("supplement config hash mismatch")
        source_root = Path(manifest["source_run"])
        if file_sha256(source_root / "dataset.sqlite") != manifest["source_database_sha256"]:
            raise ValueError("source dataset changed after supplement creation")
        mutation_config = manifest["mutation_config"]
        if json_sha256(mutation_config) != manifest["mutation_config_hash"]:
            raise ValueError("supplement mutation configuration changed")
        rows, _ = cls._source_rows(source_root)
        payload = read_checkpoint(manager.layout.latest_checkpoint)
        state = payload.get("algorithm_state")
        if not isinstance(state, dict) or state.get("runner_kind") != cls.KIND:
            raise ValueError("latest checkpoint is not a band supplement checkpoint")
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        manager.current.update(payload.get("current", {}))
        return cls(
            manager=manager,
            backend=backend,
            config=config,
            mutation_config=mutation_config,
            state=state,
            rng=rng,
            source_chromosomes=[
                Chromosome.from_iterable(json.loads(row["chromosome_json"])) for row in rows
            ],
        )

    def _archive_chromosomes(self) -> list[Chromosome]:
        return self.source_chromosomes + super()._archive_chromosomes()

    def _record_membership(self, member: dict[str, Any]) -> None:
        sample_id = member.get("sample_id")
        result = member.get("result") or {}
        if sample_id is None or result.get("status") != "completed":
            return
        existing = self.manager.database.query_all(
            "SELECT sample_id FROM supplement_membership WHERE sample_id=?", (sample_id,)
        )
        if existing:
            return
        actual_band = result.get("fitness_band")
        target_band = str(member.get("campaign_id", "TARGET-UNKNOWN")).removeprefix("TARGET-")
        targets = self.config.data["total_dataset_band_targets"]
        base = self.state["base_band_counts"]
        core_member = False
        reason = "non_target_band"
        if actual_band in TARGET_BANDS:
            current = self.manager.database.query_all(
                """SELECT COUNT(*) n FROM supplement_membership
                   WHERE run_id=? AND actual_band=? AND cnn_core_dataset_member=1""",
                (self.manager.run_id, actual_band),
            )[0]["n"]
            gap = max(0, int(targets[actual_band]) - int(base.get(actual_band, 0)))
            if int(current) < gap:
                core_member = True
                reason = "band_quota_needed"
            else:
                reason = "band_quota_full"
        self.manager.database.upsert_supplement_membership(
            sample_id=str(sample_id),
            run_id=self.manager.run_id,
            target_band=target_band,
            actual_band=str(actual_band) if actual_band is not None else None,
            physics_archive_member=True,
            cnn_core_dataset_member=core_member,
            reason=reason,
        )

    def _evaluate_member(
        self, member: dict[str, Any], *, stop_requested: Callable[[], bool]
    ) -> str:
        status = super()._evaluate_member(member, stop_requested=stop_requested)
        self._record_membership(member)
        return status

    def _evaluate_members_parallel(
        self,
        members: list[dict[str, Any]],
        *,
        stop_requested: Callable[[], bool],
    ) -> list[str]:
        statuses = super()._evaluate_members_parallel(
            members, stop_requested=stop_requested
        )
        for member in members:
            self._record_membership(member)
        return statuses

    def _mutation_config_for_campaign(self, campaign: dict[str, Any]) -> dict[str, Any]:
        target = str(campaign["target_band"])
        configured = self.config.data["scale_probabilities"][target]
        result = copy.deepcopy(self.mutation_config)
        for scale, probability in configured.items():
            result["proposal"]["scales"][scale]["probability"] = float(probability)
        return result

    def _offspring_target_for_campaign(self, campaign: dict[str, Any]) -> int:
        return int(
            self.config.data["offspring_per_band_generation"][str(campaign["target_band"])]
        )

    def _target_score(self, member: dict[str, Any], target_band: str) -> tuple[Any, ...]:
        band = _band(member)
        ratio = _ratio(member)
        target_number = _band_number(target_band)
        band_distance = abs(_band_number(band) - target_number)
        if target_band == "B7":
            return (0 if band == "B7" else 1, band_distance, -ratio, _objective(member))
        center = 0.05 + 0.1 * (target_number - 1)
        return (0 if band == target_band else 1, band_distance, abs(ratio - center), _objective(member))

    def _select_parent(self, campaign: dict[str, Any]) -> tuple[dict[str, Any], float, str]:
        population = [
            member for member in campaign["population"]
            if (member.get("result") or {}).get("status") == "completed"
        ]
        target = str(campaign["target_band"])
        eligible = set(self.config.data["eligible_parent_bands"][target])
        population = [member for member in population if _band(member) in eligible]
        if target == "B7":
            threshold = float(self.config.data.get("b7_parent_minimum_torque_ratio", 0.98))
            population = [member for member in population if _band(member) == "B7" or _ratio(member) >= threshold]
        if not population:
            raise RuntimeError(f"{campaign['campaign_id']} has no eligible parents")
        order = sorted(range(len(population)), key=lambda index: self._target_score(population[index], target))
        pressure = float(self.config.data.get("parent_rank_pressure", 2.0))
        weights = np.empty(len(population), dtype=float)
        denominator = max(1, len(population) - 1)
        for rank, index in enumerate(order):
            weights[index] = math.exp(-pressure * rank / denominator)
        probabilities = weights / weights.sum()
        index = int(self.rng.choice(len(population), p=probabilities))
        return population[index], float(probabilities[index]), f"target_{target.lower()}"

    def _select_population(self, campaign: dict[str, Any]) -> dict[str, Any]:
        size = len(campaign["population"])
        target = str(campaign["target_band"])
        pool = [
            member for member in campaign["population"] + campaign["offspring"]
            if (member.get("result") or {}).get("status") == "completed"
        ]
        if len(pool) < size:
            raise RuntimeError(f"{campaign['campaign_id']} has too few survivors")
        ordered = sorted(
            pool,
            key=lambda member: (self._target_score(member, target), str(member.get("candidate_id"))),
        )
        elite_count = min(int(self.config.data["elite_count"]), size)
        selected = ordered[:elite_count]
        remaining = [member for member in pool if member not in selected]
        while len(selected) < size:
            chosen = max(
                remaining,
                key=lambda member: (
                    min(hamming(_chromosome(member), _chromosome(other)) for other in selected),
                    tuple(-float(value) for value in self._target_score(member, target)),
                    str(member.get("candidate_id")),
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
        history = {
            "generation": int(self.state["generation"]),
            "campaign_id": campaign["campaign_id"],
            "target_band": target,
            "evaluated_candidate_ids": [row["candidate_id"] for row in campaign["offspring"]],
            "selected_candidate_ids": [row.get("candidate_id") for row in selected],
            "elite_candidate_ids": [row.get("candidate_id") for row in ordered[:elite_count]],
            "target_band_hits": sum(_band(row) == target for row in campaign["offspring"]),
        }
        campaign["population"] = selected
        campaign["offspring"] = []
        return history

    def write_campaign_status(self) -> None:
        summary = self.manager.database.progress_summary(self.manager.run_id)
        base = self.state["base_band_counts"]
        targets = self.config.data["total_dataset_band_targets"]
        supplement = summary["fitness_bands"]
        combined = {band: int(base.get(band, 0)) + int(supplement.get(band, 0)) for band in base}
        deficits = {
            band: max(0, int(targets[band]) - int(combined.get(band, 0)))
            for band in TARGET_BANDS
        }
        core_rows = self.manager.database.query_all(
            """SELECT actual_band,COUNT(*) n FROM supplement_membership
               WHERE run_id=? AND cnn_core_dataset_member=1 GROUP BY actual_band""",
            (self.manager.run_id,),
        )
        core_counts = {band: 0 for band in TARGET_BANDS}
        for row in core_rows:
            if row["actual_band"] in core_counts:
                core_counts[row["actual_band"]] = int(row["n"])
        payload = {
            "runner_kind": self.KIND,
            "generation": self.state["generation"],
            "phase": self.state["phase"],
            "campaign_index": self.state["campaign_index"],
            "proposal_count": self.state["proposal_count"],
            "new_physics_samples": summary["valid_unique_samples"],
            "target_new_physics_samples": self.config.target,
            "supplement_band_counts": supplement,
            "combined_band_counts": combined,
            "remaining_band_deficits": deficits,
            "cnn_core_supplement_counts": core_counts,
            "cnn_core_supplement_hits": sum(core_counts.values()),
            "execution_workers": int(self.state.get("execution_workers", self._worker_count())),
            "stop_reason": self.state.get("stop_reason"),
        }
        atomic_write_json(
            self.manager.layout.root / "band_supplement_status.json", payload, keep_backup=True
        )
