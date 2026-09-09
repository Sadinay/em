from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from encoding.chromosome import Chromosome, ensure_chromosome
from evaluator.base import EvaluationResult, Evaluator


@dataclass(frozen=True, slots=True)
class BatchCacheStats:
    requested: int
    unique: int
    evaluator_calls: int
    persistent_hits: int
    batch_duplicates: int


@dataclass(frozen=True, slots=True)
class BatchEvaluation:
    results: tuple[EvaluationResult, ...]
    stats: BatchCacheStats


class EvaluationCache:
    def __init__(self, values: dict[str, EvaluationResult] | None = None) -> None:
        self._values: dict[str, EvaluationResult] = dict(values or {})

    def __len__(self) -> int:
        return len(self._values)

    def evaluate_batch(
        self,
        population: Iterable[Chromosome],
        evaluator: Evaluator,
    ) -> BatchEvaluation:
        chromosomes = [ensure_chromosome(value) for value in population]
        unique: list[Chromosome] = []
        key_to_unique: dict[str, int] = {}
        inverse: list[int] = []
        for chromosome in chromosomes:
            key = chromosome.matlab_cache_key()
            if key not in key_to_unique:
                key_to_unique[key] = len(unique)
                unique.append(chromosome)
            inverse.append(key_to_unique[key])

        unique_results: list[EvaluationResult] = []
        evaluator_calls = 0
        persistent_hits = 0
        for chromosome in unique:
            key = chromosome.matlab_cache_key()
            if key in self._values:
                result = self._values[key]
                persistent_hits += 1
            else:
                result = evaluator.evaluate(chromosome)
                self._values[key] = result
                evaluator_calls += 1
            unique_results.append(result)
        results = tuple(unique_results[index] for index in inverse)
        return BatchEvaluation(
            results=results,
            stats=BatchCacheStats(
                requested=len(chromosomes),
                unique=len(unique),
                evaluator_calls=evaluator_calls,
                persistent_hits=persistent_hits,
                batch_duplicates=len(chromosomes) - len(unique),
            ),
        )

    def to_dict(self) -> dict[str, dict]:
        return {key: result.to_dict() for key, result in self._values.items()}

    @classmethod
    def from_dict(cls, values: dict[str, dict]) -> "EvaluationCache":
        return cls(
            {key: EvaluationResult.from_dict(result) for key, result in values.items()}
        )

