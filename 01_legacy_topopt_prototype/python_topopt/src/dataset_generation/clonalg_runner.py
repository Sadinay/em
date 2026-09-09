from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from clonalg.config import ClonalgConfig, clone_counts, hypermutation_probabilities, matlab_round_nonnegative
from clonalg.operators import apply_random_immigrants, initialize_population, mutate_trit
from encoding.chromosome import Chromosome

from .backend import AngleEvaluationBackend
from .checkpoint import read_checkpoint
from .orchestrator import CandidateEvaluation, DatasetRunManager


def _hamming(first: Chromosome, second: Chromosome) -> int:
    return sum(a != b for a, b in zip(first.genes, second.genes))


def _chromosome(member: dict[str, Any]) -> Chromosome:
    return Chromosome.from_iterable(member["chromosome"])


def _objective(result: dict[str, Any] | None) -> float:
    if result is None or result.get("historical_j") is None:
        return float("inf")
    value = float(result["historical_j"])
    return value if math.isfinite(value) else float("inf")


def _result_dict(result: CandidateEvaluation) -> dict[str, Any]:
    return {
        "status": result.status,
        "torque_ratio": result.torque_ratio,
        "historical_j": result.historical_j,
        "fitness_band": result.fitness_band,
        "rejection_reasons": list(result.rejection_reasons),
        "duplicate": result.duplicate,
    }


class ClonalgDatasetRunner:
    """Incremental historical CLONALG coordinator with durable mid-generation state."""

    STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        manager: DatasetRunManager,
        backend: AngleEvaluationBackend,
        generation_limit: int | None = None,
    ) -> None:
        self.manager = manager
        self.backend = backend
        raw = dict(manager.config.data["clonalg"])
        raw.pop("crossover", None)
        raw["generations"] = int(
            generation_limit or manager.config.data["limits"]["maximum_generations"]
        )
        self.config = ClonalgConfig(**raw)
        self.rng = np.random.default_rng(self.config.random_seed)
        seed = manager.seed_chromosome
        population = initialize_population(seed, self.config, self.rng)
        self.state: dict[str, Any] = {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "generation": 1,
            "generation_limit": self.config.generations,
            "generation_completed": False,
            "phase": "population_evaluation",
            "population_index": 0,
            "clone_index": 0,
            "candidate_order": 1,
            "population": self._initial_members(seed, population),
            "population_order": [],
            "clones": [],
            "clone_counts": [],
            "history": [],
            "global_best": None,
            "immigrant_positions": [],
            "immigrant_modes": [],
            "completed": False,
        }
        self.save_checkpoint()

    @classmethod
    def resume(
        cls,
        *,
        manager: DatasetRunManager,
        backend: AngleEvaluationBackend,
    ) -> "ClonalgDatasetRunner":
        payload = read_checkpoint(manager.layout.latest_checkpoint)
        if payload.get("run_id") != manager.run_id:
            raise ValueError("checkpoint belongs to another run")
        if payload.get("config_hash") != manager.config.config_hash:
            raise ValueError("checkpoint config hash mismatch")
        state = payload.get("algorithm_state")
        if not isinstance(state, dict) or state.get("schema_version") != cls.STATE_SCHEMA_VERSION:
            raise ValueError("checkpoint does not contain a compatible CLONALG state")
        runner = cls.__new__(cls)
        runner.manager = manager
        runner.backend = backend
        raw = dict(manager.config.data["clonalg"])
        raw.pop("crossover", None)
        raw["generations"] = int(state["generation_limit"])
        runner.config = ClonalgConfig(**raw)
        runner.rng = np.random.default_rng()
        runner.rng.bit_generator.state = state["rng_state"]
        runner.state = state
        manager.current.update(payload.get("current", {}))
        return runner

    def _initial_members(
        self, seed: Chromosome, population: list[Chromosome]
    ) -> list[dict[str, Any]]:
        nonseed = self.config.population_size - 1
        near = matlab_round_nonnegative(self.config.near_fraction_of_nonseed * nonseed)
        middle = matlab_round_nonnegative(self.config.middle_fraction_of_nonseed * nonseed)
        members: list[dict[str, Any]] = []
        for index, chromosome in enumerate(population):
            if index == 0:
                source, strength, operator, lineage = "seed", None, None, "L-SEED"
            elif index <= near:
                source, strength, operator, lineage = (
                    "initial_near",
                    self.config.near_mutation_probability,
                    "trit_initialization",
                    f"L-INIT-{index:02d}",
                )
            elif index <= near + middle:
                source, strength, operator, lineage = (
                    "initial_middle",
                    self.config.middle_mutation_probability,
                    "trit_initialization",
                    f"L-INIT-{index:02d}",
                )
            elif index == near + middle + 1:
                source, strength, operator, lineage = (
                    "initial_far",
                    self.config.far_mutation_probability,
                    "trit_initialization",
                    f"L-INIT-{index:02d}",
                )
            else:
                source, strength, operator, lineage = (
                    "initial_random",
                    None,
                    "full_random",
                    f"L-RANDOM-1-{index:02d}",
                )
            changed = tuple(i for i, (a, b) in enumerate(zip(seed.genes, chromosome.genes)) if a != b)
            members.append(
                {
                    "chromosome": list(chromosome.genes),
                    "source": source,
                    "parent_candidate_id": None,
                    "parent_rank": None,
                    "clone_index": None,
                    "mutation_operator": operator,
                    "mutation_strength": strength,
                    "mutation_indices": list(changed),
                    "hamming_distance_to_parent": len(changed) if index else 0,
                    "lineage_id": lineage,
                    "candidate_id": None,
                    "sample_id": None,
                    "result": None,
                }
            )
        return members

    def checkpoint_payload(self) -> dict[str, Any]:
        self.state["rng_bit_generator"] = self.rng.bit_generator.__class__.__name__
        self.state["rng_state"] = self.rng.bit_generator.state
        return self.state

    def save_checkpoint(self) -> None:
        self.manager.current["generation"] = int(self.state.get("generation", 0))
        self.manager.current["candidate_order"] = int(self.state.get("candidate_order", 0))
        self.manager.write_checkpoint(self.checkpoint_payload())

    def _candidate_from_database(
        self,
        *,
        candidate_id: str,
        sample_id: str | None,
        reasons: tuple[str, ...],
        duplicate: bool,
    ) -> CandidateEvaluation:
        if sample_id is None:
            row = self.manager.database.query_all(
                "SELECT historical_j FROM candidates WHERE candidate_id=?", (candidate_id,)
            )[0]
            return CandidateEvaluation(
                candidate_id=candidate_id,
                sample_id=None,
                status="rejected",
                torque_ratio=None,
                historical_j=float(row["historical_j"]),
                fitness_band=None,
                rejection_reasons=reasons,
            )
        sample = self.manager.database.get_sample(sample_id)
        if sample["status"] == "classified":
            result = CandidateEvaluation(
                candidate_id=candidate_id,
                sample_id=sample_id,
                status="completed",
                torque_ratio=float(sample["torque_ratio"]),
                historical_j=float(sample["historical_j"]),
                fitness_band=str(sample["fitness_band"]),
                duplicate=duplicate,
            )
            self.manager.database.update_candidate_result(
                candidate_id,
                status="completed",
                torque_ratio=result.torque_ratio,
                historical_j=result.historical_j,
                fitness_band=result.fitness_band,
            )
            return result
        return CandidateEvaluation(
            candidate_id=candidate_id,
            sample_id=sample_id,
            status=str(sample["status"]),
            torque_ratio=None,
            historical_j=None,
            fitness_band=None,
            duplicate=duplicate,
        )

    def _evaluate_member(
        self,
        member: dict[str, Any],
        *,
        stop_requested: Callable[[], bool],
    ) -> str:
        chromosome = _chromosome(member)
        if member["candidate_id"] is None:
            candidate_id, sample_id, reasons, duplicate = self.manager.register_candidate(
                chromosome,
                generation=int(self.state["generation"]),
                candidate_order=int(self.state["candidate_order"]),
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
            self.state["candidate_order"] = int(self.state["candidate_order"]) + 1
            self.save_checkpoint()
            if sample_id is None:
                member["result"] = _result_dict(
                    self._candidate_from_database(
                        candidate_id=candidate_id,
                        sample_id=None,
                        reasons=reasons,
                        duplicate=False,
                    )
                )
                self.save_checkpoint()
                return "completed"
        sample_id = member["sample_id"]
        result = self._candidate_from_database(
            candidate_id=member["candidate_id"],
            sample_id=sample_id,
            reasons=tuple(member.get("rejection_reasons", ())),
            duplicate=bool(member.get("duplicate", False)),
        )
        if result.status != "completed":
            status = self.manager.evaluate_sample(
                sample_id,
                backend=self.backend,
                chromosome=chromosome,
                stop_requested=stop_requested,
                commit_callback=self.save_checkpoint,
            )
            if status != "completed":
                return status
            result = self._candidate_from_database(
                candidate_id=member["candidate_id"],
                sample_id=sample_id,
                reasons=(),
                duplicate=bool(member.get("duplicate", False)),
            )
        member["result"] = _result_dict(result)
        self.save_checkpoint()
        return "completed"

    def _stable_order(self, members: list[dict[str, Any]]) -> list[int]:
        objectives = np.asarray([_objective(member.get("result")) for member in members])
        return [int(value) for value in np.argsort(objectives, kind="stable")]

    def _update_global_best(self, members: list[dict[str, Any]]) -> None:
        if not members:
            return
        best = members[self._stable_order(members)[0]]
        best_j = _objective(best.get("result"))
        current = self.state.get("global_best")
        if current is None or best_j < _objective(current.get("result")):
            self.state["global_best"] = {
                key: value
                for key, value in best.items()
                if key
                in {
                    "chromosome",
                    "candidate_id",
                    "sample_id",
                    "lineage_id",
                    "result",
                }
            }

    def _make_clones(self) -> None:
        population = self.state["population"]
        order = self._stable_order(population)
        self.state["population_order"] = order
        counts = clone_counts(self.config)
        probabilities = hypermutation_probabilities(self.config)
        clones: list[dict[str, Any]] = []
        global_index = 0
        for parent_rank, (count, probability) in enumerate(zip(counts, probabilities), start=1):
            parent = population[order[parent_rank - 1]]
            parent_chromosome = _chromosome(parent)
            for local_index in range(1, count + 1):
                mutation = mutate_trit(parent_chromosome, probability, self.rng)
                global_index += 1
                clones.append(
                    {
                        "chromosome": list(mutation.chromosome.genes),
                        "source": "clone",
                        "parent_candidate_id": parent["candidate_id"],
                        "parent_rank": parent_rank,
                        "clone_index": local_index,
                        "clone_global_index": global_index,
                        "mutation_operator": "trit_hypermutation",
                        "mutation_strength": probability,
                        "mutation_indices": list(mutation.changed_indices),
                        "hamming_distance_to_parent": len(mutation.changed_indices),
                        "lineage_id": parent["lineage_id"],
                        "candidate_id": None,
                        "sample_id": None,
                        "result": None,
                    }
                )
        self.state["clone_counts"] = list(counts)
        self.state["clones"] = clones
        self.state["clone_index"] = 0
        self.state["phase"] = "clone_evaluation"
        self.save_checkpoint()

    def _select_next_population(self) -> list[dict[str, Any]]:
        population = self.state["population"]
        order = self.state["population_order"]
        sorted_parents = [population[index] for index in order]
        clones = self.state["clones"]
        counts = self.state["clone_counts"]
        selected: list[dict[str, Any]] = []
        start = 0
        for rank, count in enumerate(counts, start=1):
            stop = start + int(count)
            cluster = clones[start:stop]
            clone = cluster[self._stable_order(cluster)[0]]
            parent = sorted_parents[rank - 1]
            winner = clone if _objective(clone["result"]) <= _objective(parent["result"]) else parent
            selected.append(
                {
                    "chromosome": list(winner["chromosome"]),
                    "source": "selected_clone" if winner is clone else "parent_survivor",
                    "parent_candidate_id": winner["candidate_id"],
                    "parent_rank": rank,
                    "clone_index": winner.get("clone_index"),
                    "mutation_operator": None,
                    "mutation_strength": None,
                    "mutation_indices": [],
                    "hamming_distance_to_parent": 0,
                    "lineage_id": winner["lineage_id"],
                    "candidate_id": None,
                    "sample_id": None,
                    "result": None,
                }
            )
            start = stop
        for rank, parent in enumerate(sorted_parents[len(counts) :], start=len(counts) + 1):
            selected.append(
                {
                    "chromosome": list(parent["chromosome"]),
                    "source": "uncloned_parent",
                    "parent_candidate_id": parent["candidate_id"],
                    "parent_rank": rank,
                    "clone_index": None,
                    "mutation_operator": None,
                    "mutation_strength": None,
                    "mutation_indices": [],
                    "hamming_distance_to_parent": 0,
                    "lineage_id": parent["lineage_id"],
                    "candidate_id": None,
                    "sample_id": None,
                    "result": None,
                }
            )
        return selected

    def _apply_immigrants_and_elite(self, selected: list[dict[str, Any]]) -> None:
        assert self.state["global_best"] is not None
        before = [_chromosome(member) for member in selected]
        chromosomes = list(before)
        global_best_chromosome = Chromosome.from_iterable(self.state["global_best"]["chromosome"])
        positions, modes = apply_random_immigrants(
            chromosomes,
            global_best=global_best_chromosome,
            config=self.config,
            rng=self.rng,
        )
        for position, mode in zip(positions, modes):
            chromosome = chromosomes[position]
            if mode == "BEST_MUTATION":
                parent = global_best_chromosome
                selected[position].update(
                    {
                        "source": "random_immigrant_best_mutation",
                        "parent_candidate_id": self.state["global_best"]["candidate_id"],
                        "parent_rank": None,
                        "mutation_operator": "trit_immigrant_mutation",
                        "mutation_strength": self.config.immigrant_best_mutation_probability,
                        "lineage_id": self.state["global_best"]["lineage_id"],
                    }
                )
            else:
                parent = before[position]
                selected[position].update(
                    {
                        "source": "random_immigrant_full_random",
                        "parent_candidate_id": None,
                        "parent_rank": None,
                        "mutation_operator": "full_random",
                        "mutation_strength": None,
                        "lineage_id": f"L-RANDOM-{int(self.state['generation']) + 1}-{position:02d}",
                    }
                )
            changed = [i for i, (a, b) in enumerate(zip(parent.genes, chromosome.genes)) if a != b]
            selected[position]["chromosome"] = list(chromosome.genes)
            selected[position]["mutation_indices"] = changed
            selected[position]["hamming_distance_to_parent"] = len(changed)
        elite = self.state["global_best"]
        selected[0] = {
            "chromosome": list(elite["chromosome"]),
            "source": "global_elite",
            "parent_candidate_id": elite["candidate_id"],
            "parent_rank": 1,
            "clone_index": None,
            "mutation_operator": None,
            "mutation_strength": None,
            "mutation_indices": [],
            "hamming_distance_to_parent": 0,
            "lineage_id": elite["lineage_id"],
            "candidate_id": None,
            "sample_id": None,
            "result": None,
        }
        self.state["immigrant_positions"] = list(positions)
        self.state["immigrant_modes"] = list(modes)

    def _commit_generation(self) -> None:
        population = self.state["population"]
        clones = self.state["clones"]
        self.state["history"].append(
            {
                "generation": int(self.state["generation"]),
                "population": [member["chromosome"] for member in population],
                "population_candidate_ids": [member["candidate_id"] for member in population],
                "population_historical_j": [_objective(member["result"]) for member in population],
                "sorted_indices": list(self.state["population_order"]),
                "clone_counts": list(self.state["clone_counts"]),
                "clone_candidate_ids": [member["candidate_id"] for member in clones],
                "clone_historical_j": [_objective(member["result"]) for member in clones],
                "immigrant_positions": list(self.state["immigrant_positions"]),
                "immigrant_modes": list(self.state["immigrant_modes"]),
                "global_best_candidate_id": self.state["global_best"]["candidate_id"],
                "global_best_historical_j": _objective(self.state["global_best"]["result"]),
            }
        )
        self.state["generation_completed"] = True

    def run(self, *, stop_requested: Callable[[], bool] = lambda: False) -> str:
        while not self.state["completed"]:
            if stop_requested():
                self.save_checkpoint()
                return "paused"
            stop_condition = self.manager.stop_condition()
            if stop_condition is not None:
                self.state["stop_reason"] = stop_condition
                if stop_condition == "target_reached":
                    self.state["completed"] = True
                    self.state["phase"] = "completed"
                self.save_checkpoint()
                return stop_condition
            phase = self.state["phase"]
            if phase == "population_evaluation":
                index = int(self.state["population_index"])
                if index < len(self.state["population"]):
                    status = self._evaluate_member(
                        self.state["population"][index], stop_requested=stop_requested
                    )
                    if status != "completed":
                        return status
                    self.state["population_index"] = index + 1
                    self.save_checkpoint()
                    continue
                self.state["population_order"] = self._stable_order(self.state["population"])
                self._update_global_best(self.state["population"])
                if int(self.state["generation"]) >= int(self.state["generation_limit"]):
                    self.state["clones"] = []
                    self.state["clone_counts"] = []
                    self.state["immigrant_positions"] = []
                    self.state["immigrant_modes"] = []
                    self._commit_generation()
                    self.state["completed"] = True
                    self.state["phase"] = "completed"
                    self.state["generation_completed"] = True
                    self.save_checkpoint()
                    return "completed"
                self._make_clones()
                continue
            if phase == "clone_evaluation":
                index = int(self.state["clone_index"])
                if index < len(self.state["clones"]):
                    status = self._evaluate_member(
                        self.state["clones"][index], stop_requested=stop_requested
                    )
                    if status != "completed":
                        return status
                    self.state["clone_index"] = index + 1
                    self.save_checkpoint()
                    continue
                self._update_global_best(self.state["clones"])
                selected = self._select_next_population()
                self._apply_immigrants_and_elite(selected)
                self._commit_generation()
                self.state["next_population"] = selected
                self.state["phase"] = "generation_committed"
                self.save_checkpoint()
                continue
            if phase == "generation_committed":
                self.state["population"] = self.state.pop("next_population")
                self.state["generation"] = int(self.state["generation"]) + 1
                self.state["generation_completed"] = False
                self.state["phase"] = "population_evaluation"
                self.state["population_index"] = 0
                self.state["clone_index"] = 0
                self.state["candidate_order"] = 1
                self.state["clones"] = []
                self.state["clone_counts"] = []
                self.save_checkpoint()
                continue
            if phase == "completed":
                self.state["completed"] = True
                return "completed"
            raise RuntimeError(f"unknown CLONALG phase {phase!r}")
        return "completed"
