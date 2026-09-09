from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dataset_generation.backend import AngleEvaluationBackend
from dataset_generation.checkpoint import read_checkpoint
from dataset_generation.hashing import file_sha256, json_sha256
from dataset_generation.orchestrator import CandidateEvaluation, DatasetRunManager
from dataset_generation.state import atomic_write_json
from encoding.chromosome import Chromosome

from .campaign_config import CampaignEvolutionConfig, campaign_config_from_data, load_campaign_config
from .features import hamming
from .operators import mutate_2d
from .repair import repair_topology


def _chromosome(member: dict[str, Any]) -> Chromosome:
    return Chromosome.from_iterable(member["chromosome"])


def _objective(member: dict[str, Any]) -> float:
    result = member.get("result") or {}
    value = result.get("historical_j")
    if value is None or not math.isfinite(float(value)):
        return float("inf")
    return float(value)


def _result_dict(result: CandidateEvaluation) -> dict[str, Any]:
    return {
        "status": result.status,
        "torque_ratio": result.torque_ratio,
        "historical_j": result.historical_j,
        "fitness_band": result.fitness_band,
        "duplicate": result.duplicate,
        "rejection_reasons": list(result.rejection_reasons),
    }


class ImprovedCampaignRunner:
    """Durable six-campaign mutation-only evolution for dataset production."""

    STATE_SCHEMA_VERSION = 1
    KIND = "improved_dataset_campaigns"
    METADATA_FILE = "improved_campaign_manifest.json"

    def __init__(
        self,
        *,
        manager: DatasetRunManager,
        backend: AngleEvaluationBackend,
        config: CampaignEvolutionConfig,
        mutation_config: dict[str, Any],
        state: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        self.manager = manager
        self.backend = backend
        self.config = config
        self.mutation_config = mutation_config
        self.state = state
        self.rng = rng

    def _worker_count(self) -> int:
        return max(1, min(6, int(getattr(self.backend, "max_workers", 1))))

    def _pending_offspring_batch(self) -> list[dict[str, Any]]:
        batch = self.state.get("pending_offspring_batch")
        if isinstance(batch, list):
            return batch
        legacy = self.state.get("pending_offspring")
        batch = [legacy] if isinstance(legacy, dict) else []
        self.state["pending_offspring_batch"] = batch
        self.state["pending_offspring"] = None
        return batch

    def _remaining_batch_capacity(self, stop_after_samples: int | None) -> int:
        summary = self.manager.database.progress_summary(self.manager.run_id)
        current = int(summary["valid_unique_samples"])
        limits = [max(0, int(self.config.target) - current)]
        if stop_after_samples is not None:
            limits.append(max(0, int(stop_after_samples) - current))
        return min(limits)

    @classmethod
    def create(
        cls,
        *,
        manager: DatasetRunManager,
        backend: AngleEvaluationBackend,
        parent_run: Path,
        config_path: Path,
    ) -> "ImprovedCampaignRunner":
        config = load_campaign_config(config_path)
        if config.target != manager.config.target_valid_samples:
            raise ValueError("campaign target and immutable physics run target differ")
        parent_root = Path(parent_run).resolve()
        parent_envelope = json.loads((parent_root / "run_config.json").read_text(encoding="utf-8"))
        if json_sha256(parent_envelope["config"]) != parent_envelope.get("config_hash"):
            raise ValueError("parent archive immutable config hash mismatch")
        parent_config = parent_envelope["config"]
        expected_physics_hash = parent_config["physics"]["source_config_sha256"]
        if json_sha256(manager.config.data) != expected_physics_hash:
            raise ValueError("dataset physics config differs from the parent archive physics config")
        assignments_path = parent_root / "exports" / "campaign_assignments.json"
        archive_path = parent_root / "exports" / "parent_archive.jsonl"
        assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
        archive = {
            row["parent_id"]: row
            for row in (
                json.loads(line) for line in archive_path.read_text(encoding="utf-8").splitlines() if line
            )
        }
        expected_parents = int(parent_config["parent_archive_target"])
        if len(assignments) != expected_parents or len(archive) != expected_parents:
            raise ValueError("parent archive is incomplete; finish all parents before campaign creation")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for assignment in assignments:
            record = archive[assignment["parent_id"]]
            grouped.setdefault(assignment["campaign_id"], []).append(
                {
                    "chromosome": record["chromosome_180"],
                    "source": "improved_initial_parent",
                    "parent_candidate_id": None,
                    "parent_rank": int(assignment["member_order"]),
                    "clone_index": None,
                    "mutation_operator": None,
                    "mutation_strength": None,
                    "mutation_indices": [],
                    "hamming_distance_to_parent": 0,
                    "lineage_id": record["lineage_id"],
                    "archive_parent_id": record["parent_id"],
                    "topology_cluster": int(assignment["topology_cluster"]),
                    "candidate_id": None,
                    "sample_id": None,
                    "result": None,
                    "proposal_order": None,
                }
            )
        campaign_count = int(parent_config["campaign_count"])
        per_campaign = int(parent_config["parents_per_campaign"])
        if len(grouped) != campaign_count or any(len(rows) != per_campaign for rows in grouped.values()):
            raise ValueError("campaign assignment is not balanced")
        campaigns = [
            {"campaign_id": campaign_id, "population": grouped[campaign_id], "offspring": []}
            for campaign_id in sorted(grouped)
        ]
        rng = np.random.default_rng(config.random_seed)
        state: dict[str, Any] = {
            "schema_version": cls.STATE_SCHEMA_VERSION,
            "runner_kind": cls.KIND,
            "generation": 0,
            "generation_completed": False,
            "phase": "initial_evaluation",
            "campaign_index": 0,
            "member_index": 0,
            "candidate_order": 1,
            "proposal_count": 0,
            "pending_offspring": None,
            "pending_offspring_batch": [],
            "campaigns": campaigns,
            "history": [],
            "completed": False,
            "stop_reason": None,
        }
        manifest = {
            "schema_version": 1,
            "runner_kind": cls.KIND,
            "campaign_config_hash": config.config_hash,
            "campaign_config": config.data,
            "parent_run": str(parent_root),
            "parent_config_hash": parent_envelope["config_hash"],
            "parent_archive_sha256": file_sha256(archive_path),
            "campaign_assignments_sha256": file_sha256(assignments_path),
            "dataset_config_hash": manager.config.config_hash,
        }
        atomic_write_json(manager.layout.root / cls.METADATA_FILE, manifest)
        runner = cls(
            manager=manager,
            backend=backend,
            config=config,
            mutation_config=parent_config,
            state=state,
            rng=rng,
        )
        runner.save_checkpoint()
        runner.write_campaign_status()
        return runner

    @classmethod
    def resume(
        cls, *, manager: DatasetRunManager, backend: AngleEvaluationBackend
    ) -> "ImprovedCampaignRunner":
        manifest_path = manager.layout.root / cls.METADATA_FILE
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("runner_kind") != cls.KIND:
            raise ValueError("run is not an improved campaign run")
        if manifest.get("dataset_config_hash") != manager.config.config_hash:
            raise ValueError("campaign manifest and dataset config differ")
        config = campaign_config_from_data(manifest["campaign_config"], manifest_path)
        if config.config_hash != manifest.get("campaign_config_hash"):
            raise ValueError("campaign config hash mismatch")
        parent_root = Path(manifest["parent_run"])
        archive_path = parent_root / "exports" / "parent_archive.jsonl"
        assignments_path = parent_root / "exports" / "campaign_assignments.json"
        if file_sha256(archive_path) != manifest["parent_archive_sha256"]:
            raise ValueError("parent archive changed after campaign creation")
        if file_sha256(assignments_path) != manifest["campaign_assignments_sha256"]:
            raise ValueError("campaign assignments changed after campaign creation")
        parent_envelope = json.loads((parent_root / "run_config.json").read_text(encoding="utf-8"))
        if parent_envelope.get("config_hash") != manifest["parent_config_hash"]:
            raise ValueError("parent run config changed after campaign creation")
        payload = read_checkpoint(manager.layout.latest_checkpoint)
        state = payload.get("algorithm_state")
        if not isinstance(state, dict) or state.get("runner_kind") != cls.KIND:
            raise ValueError("latest checkpoint is not an improved campaign checkpoint")
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        manager.current.update(payload.get("current", {}))
        return cls(
            manager=manager,
            backend=backend,
            config=config,
            mutation_config=parent_envelope["config"],
            state=state,
            rng=rng,
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        self.state["rng_state"] = self.rng.bit_generator.state
        self.state["rng_bit_generator"] = self.rng.bit_generator.__class__.__name__
        self.state["population"] = [
            member["chromosome"]
            for campaign in self.state["campaigns"]
            for member in campaign["population"]
        ]
        return self.state

    def save_checkpoint(self) -> None:
        self.manager.current["generation"] = int(self.state["generation"])
        self.manager.current["candidate_order"] = int(self.state["candidate_order"])
        self.manager.write_checkpoint(self.checkpoint_payload())

    def write_campaign_status(self) -> None:
        summary = self.manager.database.progress_summary(self.manager.run_id)
        proposal_rows = self.manager.database.query_all(
            """SELECT admission_status,COUNT(*) n FROM improved_proposals
               WHERE run_id=? GROUP BY admission_status""",
            (self.manager.run_id,),
        )
        best_by_campaign: dict[str, float | None] = {}
        for campaign in self.state["campaigns"]:
            values = [_objective(member) for member in campaign["population"]]
            finite = [value for value in values if math.isfinite(value)]
            best_by_campaign[campaign["campaign_id"]] = min(finite) if finite else None
        payload = {
            "runner_kind": self.KIND,
            "generation": self.state["generation"],
            "phase": self.state["phase"],
            "campaign_index": self.state["campaign_index"],
            "proposal_count": self.state["proposal_count"],
            "proposal_status_counts": {row["admission_status"]: int(row["n"]) for row in proposal_rows},
            "valid_unique_samples": summary["valid_unique_samples"],
            "target_valid_unique_samples": self.config.target,
            "fitness_bands": summary["fitness_bands"],
            "best_historical_j_by_campaign": best_by_campaign,
            "completed_campaign_generations": len(self.state["history"]),
            "execution_workers": int(self.state.get("execution_workers", self._worker_count())),
            "stop_reason": self.state.get("stop_reason"),
        }
        atomic_write_json(self.manager.layout.root / "improved_campaign_status.json", payload, keep_backup=True)

    def _candidate_from_database(self, member: dict[str, Any]) -> CandidateEvaluation:
        candidate_id = str(member["candidate_id"])
        sample_id = member.get("sample_id")
        if sample_id is None:
            row = self.manager.database.query_all(
                "SELECT historical_j,rejection_reasons_json FROM candidates WHERE candidate_id=?",
                (candidate_id,),
            )[0]
            return CandidateEvaluation(
                candidate_id, None, "rejected", None, float(row["historical_j"]), None,
                tuple(json.loads(row["rejection_reasons_json"])), False,
            )
        sample = self.manager.database.get_sample(sample_id)
        if sample["status"] != "classified":
            return CandidateEvaluation(candidate_id, sample_id, str(sample["status"]), None, None, None)
        result = CandidateEvaluation(
            candidate_id, sample_id, "completed", float(sample["torque_ratio"]),
            float(sample["historical_j"]), str(sample["fitness_band"]), (),
            bool(member.get("duplicate", False)),
        )
        self.manager.database.update_candidate_result(
            candidate_id,
            status="completed",
            torque_ratio=result.torque_ratio,
            historical_j=result.historical_j,
            fitness_band=result.fitness_band,
        )
        return result

    def _register_or_recover(self, member: dict[str, Any]) -> None:
        if member["candidate_id"] is not None:
            return
        generation = int(member.get("generation", self.state["generation"]))
        order = int(member.get("candidate_order", self.state["candidate_order"]))
        existing = self.manager.database.query_all(
            "SELECT * FROM candidates WHERE run_id=? AND generation=? AND candidate_order=?",
            (self.manager.run_id, generation, order),
        )
        if existing:
            row = existing[0]
            if row["chromosome_hash"] != _chromosome(member).sha256():
                raise ValueError("candidate recovery key points to another chromosome")
            member["candidate_id"] = row["candidate_id"]
            member["sample_id"] = row["duplicate_of_sample_id"]
            if member["sample_id"] is None:
                samples = self.manager.database.query_all(
                    "SELECT sample_id FROM samples WHERE first_candidate_id=?", (row["candidate_id"],)
                )
                member["sample_id"] = samples[0]["sample_id"] if samples else None
            member["duplicate"] = row["validity_status"] == "duplicate"
        else:
            candidate_id, sample_id, reasons, duplicate = self.manager.register_candidate(
                _chromosome(member),
                generation=generation,
                candidate_order=order,
                source=member["source"],
                parent_candidate_id=member.get("parent_candidate_id"),
                parent_rank=member.get("parent_rank"),
                clone_index=member.get("clone_index"),
                mutation_operator=member.get("mutation_operator"),
                mutation_strength=member.get("mutation_strength"),
                mutation_indices=member.get("mutation_indices", ()),
                hamming_distance_to_parent=member.get("hamming_distance_to_parent"),
                lineage_id=member.get("lineage_id"),
            )
            member["candidate_id"] = candidate_id
            member["sample_id"] = sample_id
            member["duplicate"] = duplicate
            member["rejection_reasons"] = list(reasons)
        if member.get("proposal_order") is not None:
            self.manager.database.link_improved_proposal(
                int(member["proposal_order"]), str(member["candidate_id"])
            )
        self.state["candidate_order"] = max(int(self.state["candidate_order"]), order + 1)
        self.save_checkpoint()

    def _evaluate_member(
        self, member: dict[str, Any], *, stop_requested: Callable[[], bool]
    ) -> str:
        self._register_or_recover(member)
        if member.get("sample_id") is None:
            member["result"] = _result_dict(self._candidate_from_database(member))
            self.save_checkpoint()
            return "completed"
        result = self._candidate_from_database(member)
        if result.status != "completed":
            status = self.manager.evaluate_sample(
                str(member["sample_id"]), backend=self.backend, chromosome=_chromosome(member),
                stop_requested=stop_requested, commit_callback=self.save_checkpoint,
            )
            if status != "completed":
                if status == "failed":
                    sample = self.manager.database.get_sample(str(member["sample_id"]))
                    self.manager.database.update_candidate_status(
                        str(member["candidate_id"]), "failed"
                    )
                    member["result"] = {
                        "status": "physical_failed",
                        "torque_ratio": None,
                        "historical_j": None,
                        "fitness_band": None,
                        "duplicate": False,
                        "rejection_reasons": [
                            f"{sample['error_type']}: {sample['error_message']}"
                        ],
                    }
                    self.save_checkpoint()
                    return "physical_failed"
                return status
            result = self._candidate_from_database(member)
        member["result"] = _result_dict(result)
        self.save_checkpoint()
        return "completed"

    def _evaluate_members_parallel(
        self,
        members: list[dict[str, Any]],
        *,
        stop_requested: Callable[[], bool],
    ) -> list[str]:
        if len(members) <= 1 or self._worker_count() <= 1:
            return [
                self._evaluate_member(member, stop_requested=stop_requested)
                for member in members
            ]

        request_members: list[dict[str, Any]] = []
        for member in members:
            self._register_or_recover(member)
            if member.get("sample_id") is None:
                member["result"] = _result_dict(self._candidate_from_database(member))
                continue
            result = self._candidate_from_database(member)
            if result.status == "completed":
                member["result"] = _result_dict(result)
            else:
                request_members.append(member)

        statuses_by_sample: dict[str, str] = {}
        if request_members:
            statuses_by_sample = self.manager.evaluate_samples_parallel(
                [
                    (str(member["sample_id"]), _chromosome(member))
                    for member in request_members
                ],
                backend=self.backend,
                max_workers=self._worker_count(),
                stop_requested=stop_requested,
                commit_callback=self.save_checkpoint,
            )

        statuses: list[str] = []
        for member in members:
            sample_id = member.get("sample_id")
            if sample_id is None:
                statuses.append("completed")
                continue
            status = statuses_by_sample.get(str(sample_id))
            if status is None:
                result = self._candidate_from_database(member)
                member["result"] = _result_dict(result)
                statuses.append("completed")
                continue
            if status == "completed":
                result = self._candidate_from_database(member)
                member["result"] = _result_dict(result)
                statuses.append("completed")
                continue
            if status == "failed":
                sample = self.manager.database.get_sample(str(sample_id))
                self.manager.database.update_candidate_status(
                    str(member["candidate_id"]), "failed"
                )
                member["result"] = {
                    "status": "physical_failed",
                    "torque_ratio": None,
                    "historical_j": None,
                    "fitness_band": None,
                    "duplicate": False,
                    "rejection_reasons": [
                        f"{sample['error_type']}: {sample['error_message']}"
                    ],
                }
                statuses.append("physical_failed")
                continue
            statuses.append(status)
        self.save_checkpoint()
        return statuses

    def _archive_chromosomes(self) -> list[Chromosome]:
        rows = self.manager.database.query_all(
            """SELECT chromosome_json FROM samples WHERE run_id=? AND sample_kind='candidate'
               AND chromosome_json IS NOT NULL""",
            (self.manager.run_id,),
        )
        return [Chromosome.from_iterable(json.loads(row["chromosome_json"])) for row in rows]

    def _select_parent(self, campaign: dict[str, Any]) -> tuple[dict[str, Any], float, str]:
        population = [
            member
            for member in campaign["population"]
            if (member.get("result") or {}).get("status") == "completed"
        ]
        if not population:
            raise RuntimeError(f"{campaign['campaign_id']} has no physically valid parents")
        selection = self.config.data.get("parent_selection", {})
        exploitation_probability = float(selection.get("objective_exploitation_probability", 0.0))
        if float(self.rng.random()) < exploitation_probability:
            pressure = float(selection.get("objective_rank_pressure", 2.0))
            order = sorted(range(len(population)), key=lambda index: (_objective(population[index]), index))
            weights = np.empty(len(population), dtype=float)
            denominator = max(1, len(population) - 1)
            for rank, index in enumerate(order):
                weights[index] = math.exp(-pressure * rank / denominator)
            probabilities = weights / weights.sum()
            selection_mode = "objective_exploitation"
        else:
            guidance = selection.get("band_deficit_guidance", True)
            if not guidance:
                probabilities = np.full(len(population), 1.0 / len(population))
                selection_mode = "uniform_exploration"
            else:
                summary = self.manager.database.progress_summary(self.manager.run_id)
                targets = self.manager.config.data["fitness_bands"]["targets"]
                cap = float(selection.get("maximum_band_weight", 4.0))
                weights = []
                for member in population:
                    band = (member.get("result") or {}).get("fitness_band")
                    if band in targets:
                        target = max(1, int(targets[band]))
                        deficit = max(0, target - int(summary["fitness_bands"][band])) / target
                        weights.append(min(cap, 1.0 + deficit * (cap - 1.0)))
                    else:
                        weights.append(1.0)
                probabilities = np.asarray(weights, dtype=float)
                probabilities /= probabilities.sum()
                selection_mode = "band_deficit_exploration"
        index = int(self.rng.choice(len(population), p=probabilities))
        return population[index], float(probabilities[index]), selection_mode

    def _mutation_config_for_campaign(self, campaign: dict[str, Any]) -> dict[str, Any]:
        """Return the mutation configuration used for one campaign proposal.

        The historical improved campaign uses one immutable configuration for
        every campaign.  Specialized dataset samplers may override this hook
        while retaining the durable proposal/evaluation machinery.
        """

        del campaign
        return self.mutation_config

    def _offspring_target_for_campaign(self, campaign: dict[str, Any]) -> int:
        del campaign
        return int(self.config.data["offspring_per_campaign_generation"])

    def _make_proposal(self, campaign: dict[str, Any]) -> dict[str, Any] | None:
        parent, selection_probability, selection_mode = self._select_parent(campaign)
        parent_chromosome = _chromosome(parent)
        mutation = mutate_2d(
            parent_chromosome,
            self._mutation_config_for_campaign(campaign),
            self.rng,
        )
        repair = repair_topology(mutation.chromosome, self.mutation_config)
        repaired = repair.repaired
        parent_distance = hamming(parent_chromosome, repaired)
        archive = self._archive_chromosomes()
        nearest = min((hamming(repaired, row) for row in archive), default=None)
        limits = self.config.data["hamming_admission"][mutation.scale]
        reasons = list(repair.rejection_reasons)
        if repair.status not in {"unchanged_legal", "repaired"}:
            reasons.append("REPAIR_REJECTED")
        if nearest is not None and nearest < int(limits["minimum_archive_distance"]):
            reasons.append("ARCHIVE_HAMMING_TOO_SMALL")
        if parent_distance > int(limits["maximum_parent_distance"]):
            reasons.append("PARENT_HAMMING_TOO_LARGE")
        admission = "accepted" if not reasons else "rejected"
        self.state["proposal_count"] = int(self.state["proposal_count"]) + 1
        proposal_order = int(self.state["proposal_count"])
        metadata = {
            **mutation.metadata,
            "requested_cells": mutation.requested_cells,
            "changed_indices_before_repair": list(mutation.changed_indices),
            "repair_iterations": repair.iterations,
            "repair_actions": list(repair.actions),
            "selection_probability": selection_probability,
            "parent_selection_mode": selection_mode,
            "execution_workers": self._worker_count(),
        }
        if parent.get("external_candidate_id") is not None:
            metadata["external_parent_candidate_id"] = parent["external_candidate_id"]
            metadata["external_parent_sample_id"] = parent.get("external_sample_id")
        if campaign.get("target_band") is not None:
            metadata["target_band"] = campaign["target_band"]
        self.manager.database.insert_improved_proposal(
            {
                "run_id": self.manager.run_id,
                "proposal_order": proposal_order,
                "generation": self.state["generation"],
                "campaign_id": campaign["campaign_id"],
                "parent_candidate_id": parent.get("candidate_id"),
                "operator": mutation.operator,
                "scale": mutation.scale,
                "raw_chromosome": list(mutation.chromosome.genes),
                "raw_hash": mutation.chromosome.sha256(),
                "repaired_chromosome": list(repaired.genes),
                "repaired_hash": repaired.sha256(),
                "repair_status": repair.status,
                "repair_hamming": repair.repair_hamming,
                "parent_hamming": parent_distance,
                "nearest_archive_hamming": nearest,
                "admission_status": admission,
                "rejection_reasons": reasons,
                "metadata": metadata,
            }
        )
        if reasons:
            self.save_checkpoint()
            return None
        member = {
            "chromosome": list(repaired.genes),
            "source": "improved_campaign_offspring",
            "parent_candidate_id": parent.get("candidate_id"),
            "parent_rank": None,
            "clone_index": (
                len(campaign["offspring"]) + len(self._pending_offspring_batch()) + 1
            ),
            "mutation_operator": f"{mutation.operator}:{mutation.scale}",
            "mutation_strength": mutation.requested_cells / 180.0,
            "mutation_indices": [
                index for index, (a, b) in enumerate(zip(parent_chromosome.genes, repaired.genes)) if a != b
            ],
            "hamming_distance_to_parent": parent_distance,
            "lineage_id": parent["lineage_id"],
            "candidate_id": None,
            "sample_id": None,
            "result": None,
            "proposal_order": proposal_order,
            "generation": int(self.state["generation"]),
            "candidate_order": int(self.state["candidate_order"]),
            "campaign_id": campaign["campaign_id"],
            "selection_probability": selection_probability,
            "parent_selection_mode": selection_mode,
        }
        self._pending_offspring_batch().append(member)
        self.state["pending_offspring"] = None
        self.save_checkpoint()
        return member

    def _select_population(self, campaign: dict[str, Any]) -> dict[str, Any]:
        size = len(campaign["population"])
        previous_best = min(_objective(row) for row in campaign["population"])
        pool = [
            row
            for row in campaign["population"] + campaign["offspring"]
            if (row.get("result") or {}).get("status") == "completed"
        ]
        if len(pool) < size:
            raise RuntimeError(
                f"{campaign['campaign_id']} has only {len(pool)} physical survivors for {size} slots"
            )
        ordered = sorted(pool, key=lambda row: (_objective(row), str(row.get("candidate_id"))))
        elite_count = min(int(self.config.data["elite_count"]), size)
        selected = ordered[:elite_count]
        remaining = [row for row in pool if row not in selected]
        while len(selected) < size:
            chosen = max(
                remaining,
                key=lambda row: (
                    min(hamming(_chromosome(row), _chromosome(other)) for other in selected),
                    -_objective(row),
                    str(row.get("candidate_id")),
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
        history = {
            "generation": int(self.state["generation"]),
            "campaign_id": campaign["campaign_id"],
            "evaluated_candidate_ids": [row["candidate_id"] for row in campaign["offspring"]],
            "selected_candidate_ids": [row["candidate_id"] for row in selected],
            "elite_candidate_ids": [row["candidate_id"] for row in ordered[:elite_count]],
            "previous_best_historical_j": previous_best,
            "best_historical_j": _objective(ordered[0]),
        }
        campaign["population"] = selected
        campaign["offspring"] = []
        return history

    def run(
        self,
        *,
        stop_requested: Callable[[], bool] = lambda: False,
        stop_after_samples: int | None = None,
    ) -> str:
        self.state["execution_workers"] = self._worker_count()
        self._pending_offspring_batch()
        while not self.state["completed"]:
            if stop_requested():
                self.state["stop_reason"] = "user_pause"
                self.save_checkpoint()
                self.write_campaign_status()
                return "paused"
            summary = self.manager.database.progress_summary(self.manager.run_id)
            if stop_after_samples is not None and int(summary["valid_unique_samples"]) >= int(stop_after_samples):
                self.state["stop_reason"] = f"pilot_sample_limit_{int(stop_after_samples)}"
                self.save_checkpoint()
                self.write_campaign_status()
                return "pilot_stop"
            stop = self.manager.stop_condition()
            if stop == "target_reached":
                completed_generations = max(0, int(self.state["generation"]) - 1)
                if completed_generations < int(self.config.data["minimum_completed_generations"]):
                    stop = None
            if stop is not None:
                self.state["stop_reason"] = stop
                if stop == "target_reached":
                    self.state["completed"] = True
                    self.state["phase"] = "completed"
                self.save_checkpoint()
                self.write_campaign_status()
                return stop
            if int(self.state["proposal_count"]) >= int(self.config.data["maximum_proposals"]):
                self.state["stop_reason"] = "maximum_proposals"
                self.save_checkpoint()
                self.write_campaign_status()
                return "maximum_proposals"

            if self.state["phase"] == "initial_evaluation":
                campaign_index = int(self.state["campaign_index"])
                member_index = int(self.state["member_index"])
                if campaign_index >= len(self.state["campaigns"]):
                    self.state.update(
                        generation=1, phase="offspring_generation", campaign_index=0,
                        member_index=0, generation_completed=False,
                    )
                    self.save_checkpoint()
                    continue
                campaign = self.state["campaigns"][campaign_index]
                if member_index >= len(campaign["population"]):
                    self.state["campaign_index"] = campaign_index + 1
                    self.state["member_index"] = 0
                    self.save_checkpoint()
                    continue
                capacity = max(1, self._remaining_batch_capacity(stop_after_samples))
                batch = campaign["population"][
                    member_index : member_index + min(self._worker_count(), capacity)
                ]
                for member in batch:
                    member.setdefault("generation", 0)
                statuses = self._evaluate_members_parallel(
                    batch, stop_requested=stop_requested
                )
                if any(status not in {"completed", "physical_failed"} for status in statuses):
                    self.write_campaign_status()
                    return next(
                        status
                        for status in statuses
                        if status not in {"completed", "physical_failed"}
                    )
                for status in statuses:
                    if status == "physical_failed":
                        self.state["consecutive_physical_failures"] = int(
                            self.state.get("consecutive_physical_failures", 0)
                        ) + 1
                    else:
                        self.state["consecutive_physical_failures"] = 0
                self.state["member_index"] = member_index + len(batch)
                self.save_checkpoint()
                self.write_campaign_status()
                if int(self.state.get("consecutive_physical_failures", 0)) >= int(
                    self.manager.config.data["recovery"]["max_consecutive_femm_failures"]
                ):
                    self.state["stop_reason"] = "maximum_consecutive_femm_failures"
                    self.save_checkpoint()
                    return "failure_guard"
                continue

            if self.state["phase"] == "offspring_generation":
                if int(self.state["generation"]) > int(self.config.data["maximum_generations"]):
                    self.state["stop_reason"] = "maximum_generations"
                    self.save_checkpoint()
                    self.write_campaign_status()
                    return "maximum_generations"
                campaign_index = int(self.state["campaign_index"])
                if campaign_index >= len(self.state["campaigns"]):
                    self.state["generation_completed"] = True
                    self.save_checkpoint()
                    self.state["generation"] = int(self.state["generation"]) + 1
                    self.state["campaign_index"] = 0
                    self.state["generation_completed"] = False
                    self.save_checkpoint()
                    continue
                campaign = self.state["campaigns"][campaign_index]
                target = self._offspring_target_for_campaign(campaign)
                if len(campaign["offspring"]) >= target:
                    self.state["history"].append(self._select_population(campaign))
                    self.state["campaign_index"] = campaign_index + 1
                    self.save_checkpoint()
                    self.write_campaign_status()
                    continue
                batch = self._pending_offspring_batch()
                batch_target = min(
                    self._worker_count(),
                    target - len(campaign["offspring"]),
                    max(1, self._remaining_batch_capacity(stop_after_samples)),
                )
                while len(batch) < batch_target:
                    member = self._make_proposal(campaign)
                    if member is None:
                        continue
                    # Register before proposing the next child so full-archive
                    # Hamming admission also sees every pending chromosome.
                    self._register_or_recover(member)
                statuses = self._evaluate_members_parallel(
                    list(batch), stop_requested=stop_requested
                )
                if any(status not in {"completed", "physical_failed"} for status in statuses):
                    self.write_campaign_status()
                    return next(
                        status
                        for status in statuses
                        if status not in {"completed", "physical_failed"}
                    )
                for member, status in zip(list(batch), statuses):
                    if status == "physical_failed":
                        self.state["consecutive_physical_failures"] = int(
                            self.state.get("consecutive_physical_failures", 0)
                        ) + 1
                        continue
                    self.state["consecutive_physical_failures"] = 0
                    if not member.get("duplicate"):
                        campaign["offspring"].append(member)
                batch.clear()
                self.state["pending_offspring"] = None
                self.save_checkpoint()
                self.write_campaign_status()
                if int(self.state.get("consecutive_physical_failures", 0)) >= int(
                    self.manager.config.data["recovery"]["max_consecutive_femm_failures"]
                ):
                    self.state["stop_reason"] = "maximum_consecutive_femm_failures"
                    self.save_checkpoint()
                    return "failure_guard"
                continue
            raise ValueError(f"unknown improved campaign phase {self.state['phase']}")
        return "target_reached"
