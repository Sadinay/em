from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from encoding.chromosome import CHROMOSOME_LENGTH, Chromosome, ensure_chromosome
from evaluator.base import Evaluator

from .cache import BatchCacheStats, EvaluationCache
from .checkpoint import read_checkpoint, write_checkpoint_atomic
from .config import (
    ClonalgConfig,
    clone_counts,
    hypermutation_probabilities,
)
from .history import GenerationRecord
from .operators import (
    apply_random_immigrants,
    initialize_population,
    mutate_trit,
    select_parent_or_best_clone,
    stable_fitness_order,
)


@dataclass(frozen=True, slots=True)
class ClonalgRunResult:
    best_chromosome: Chromosome
    best_objective: float
    history: tuple[GenerationRecord, ...]
    completed: bool
    next_generation: int
    cache_size: int


def _stats_dict(stats: BatchCacheStats) -> dict[str, int]:
    return {
        "requested": stats.requested,
        "unique": stats.unique,
        "evaluator_calls": stats.evaluator_calls,
        "persistent_hits": stats.persistent_hits,
        "batch_duplicates": stats.batch_duplicates,
    }


class ClonalgEngine:
    def __init__(
        self,
        *,
        seed_chromosome: Chromosome | Sequence[int],
        evaluator: Evaluator,
        config: ClonalgConfig = ClonalgConfig(),
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.rng = np.random.default_rng(config.random_seed)
        self.population = initialize_population(
            ensure_chromosome(seed_chromosome), config, self.rng
        )
        self.cache = EvaluationCache()
        self.history: list[GenerationRecord] = []
        self.global_best: Chromosome | None = None
        self.global_best_objective = float("inf")
        self.next_generation = 1
        self.completed = False

    def _make_clones(
        self, sorted_population: list[Chromosome]
    ) -> tuple[list[Chromosome], tuple[int, ...]]:
        counts = clone_counts(self.config)
        probabilities = hypermutation_probabilities(self.config)
        clones: list[Chromosome] = []
        for parent, count, probability in zip(
            sorted_population[: len(counts)], counts, probabilities
        ):
            for _ in range(count):
                clones.append(mutate_trit(parent, probability, self.rng).chromosome)
        return clones, counts

    def _result(self) -> ClonalgRunResult:
        if self.global_best is None:
            raise RuntimeError("the engine has not evaluated a generation")
        return ClonalgRunResult(
            best_chromosome=self.global_best,
            best_objective=float(self.global_best_objective),
            history=tuple(self.history),
            completed=self.completed,
            next_generation=self.next_generation,
            cache_size=len(self.cache),
        )

    def run(
        self,
        *,
        stop_after_generation: int | None = None,
        checkpoint_path: Path | None = None,
    ) -> ClonalgRunResult:
        if self.completed:
            return self._result()
        while self.next_generation <= self.config.generations:
            generation = self.next_generation
            population_batch = self.cache.evaluate_batch(
                self.population, self.evaluator
            )
            objectives = np.asarray(
                [result.objective for result in population_batch.results], dtype=float
            )
            order = stable_fitness_order(objectives)
            sorted_population = [self.population[int(index)] for index in order]
            sorted_objectives = objectives[order]
            current_best = sorted_population[0]
            current_best_objective = float(sorted_objectives[0])
            if current_best_objective < self.global_best_objective:
                self.global_best = current_best
                self.global_best_objective = current_best_objective

            counts: tuple[int, ...] = ()
            clone_objective_values: tuple[float, ...] = ()
            immigrant_positions: tuple[int, ...] = ()
            immigrant_modes: tuple[str, ...] = ()
            clone_stats: dict[str, int] = {}

            if generation < self.config.generations:
                clones, counts = self._make_clones(sorted_population)
                clone_batch = self.cache.evaluate_batch(clones, self.evaluator)
                clone_objectives = np.asarray(
                    [result.objective for result in clone_batch.results], dtype=float
                )
                clone_objective_values = tuple(float(value) for value in clone_objectives)
                clone_stats = _stats_dict(clone_batch.stats)
                minimum_clone_index = int(np.argmin(clone_objectives))
                if clone_objectives[minimum_clone_index] < self.global_best_objective:
                    self.global_best = clones[minimum_clone_index]
                    self.global_best_objective = float(
                        clone_objectives[minimum_clone_index]
                    )
                next_population = select_parent_or_best_clone(
                    sorted_population,
                    sorted_objectives,
                    clones,
                    clone_objectives,
                    counts,
                )
                assert self.global_best is not None
                immigrant_positions, immigrant_modes = apply_random_immigrants(
                    next_population,
                    global_best=self.global_best,
                    config=self.config,
                    rng=self.rng,
                )
            else:
                next_population = self.population

            assert self.global_best is not None
            record = GenerationRecord(
                generation=generation,
                population=tuple(chromosome.genes for chromosome in self.population),
                objectives=tuple(float(value) for value in objectives),
                sorted_indices=tuple(int(index) for index in order),
                current_best=current_best.genes,
                current_best_objective=current_best_objective,
                global_best=self.global_best.genes,
                global_best_objective=float(self.global_best_objective),
                clone_counts=counts,
                clone_objectives=clone_objective_values,
                immigrant_positions=immigrant_positions,
                immigrant_modes=immigrant_modes,
                population_cache_stats=_stats_dict(population_batch.stats),
                clone_cache_stats=clone_stats,
            )
            self.history.append(record)
            self.next_generation = generation + 1
            if generation >= self.config.generations:
                self.completed = True
            else:
                self.population = next_population

            if checkpoint_path is not None:
                self.save_checkpoint(checkpoint_path)
            if (
                stop_after_generation is not None
                and generation >= stop_after_generation
            ):
                break
        return self._result()

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "next_generation": self.next_generation,
            "completed": self.completed,
            "population": [list(chromosome.genes) for chromosome in self.population],
            "global_best": list(self.global_best.genes) if self.global_best else None,
            "global_best_objective": self.global_best_objective,
            "history": [record.to_dict() for record in self.history],
            "cache": self.cache.to_dict(),
            "rng_bit_generator": self.rng.bit_generator.__class__.__name__,
            "rng_state": self.rng.bit_generator.state,
        }

    def save_checkpoint(self, path: Path) -> None:
        write_checkpoint_atomic(Path(path), self.checkpoint_payload())

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        evaluator: Evaluator,
    ) -> "ClonalgEngine":
        payload = read_checkpoint(Path(path))
        config = ClonalgConfig(**payload["config"])
        engine = cls.__new__(cls)
        engine.config = config
        engine.evaluator = evaluator
        engine.rng = np.random.default_rng()
        if payload.get("rng_bit_generator") != engine.rng.bit_generator.__class__.__name__:
            raise ValueError("checkpoint uses an unsupported NumPy bit generator")
        engine.rng.bit_generator.state = payload["rng_state"]
        engine.population = [
            Chromosome.from_iterable(row) for row in payload["population"]
        ]
        engine.cache = EvaluationCache.from_dict(payload["cache"])
        engine.history = [
            GenerationRecord.from_dict(value) for value in payload["history"]
        ]
        best = payload.get("global_best")
        engine.global_best = Chromosome.from_iterable(best) if best is not None else None
        engine.global_best_objective = float(payload["global_best_objective"])
        engine.next_generation = int(payload["next_generation"])
        engine.completed = bool(payload["completed"])
        return engine
