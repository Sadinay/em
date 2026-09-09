from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from encoding.chromosome import Chromosome


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation: int
    population: tuple[tuple[int, ...], ...]
    objectives: tuple[float, ...]
    sorted_indices: tuple[int, ...]
    current_best: tuple[int, ...]
    current_best_objective: float
    global_best: tuple[int, ...]
    global_best_objective: float
    clone_counts: tuple[int, ...]
    clone_objectives: tuple[float, ...]
    immigrant_positions: tuple[int, ...]
    immigrant_modes: tuple[str, ...]
    population_cache_stats: dict[str, int]
    clone_cache_stats: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "population": [list(row) for row in self.population],
            "objectives": list(self.objectives),
            "sorted_indices": list(self.sorted_indices),
            "current_best": list(self.current_best),
            "current_best_objective": self.current_best_objective,
            "global_best": list(self.global_best),
            "global_best_objective": self.global_best_objective,
            "clone_counts": list(self.clone_counts),
            "clone_objectives": list(self.clone_objectives),
            "immigrant_positions": list(self.immigrant_positions),
            "immigrant_modes": list(self.immigrant_modes),
            "population_cache_stats": self.population_cache_stats,
            "clone_cache_stats": self.clone_cache_stats,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationRecord":
        return cls(
            generation=int(value["generation"]),
            population=tuple(tuple(int(x) for x in row) for row in value["population"]),
            objectives=tuple(float(x) for x in value["objectives"]),
            sorted_indices=tuple(int(x) for x in value["sorted_indices"]),
            current_best=tuple(int(x) for x in value["current_best"]),
            current_best_objective=float(value["current_best_objective"]),
            global_best=tuple(int(x) for x in value["global_best"]),
            global_best_objective=float(value["global_best_objective"]),
            clone_counts=tuple(int(x) for x in value.get("clone_counts", ())),
            clone_objectives=tuple(float(x) for x in value.get("clone_objectives", ())),
            immigrant_positions=tuple(int(x) for x in value.get("immigrant_positions", ())),
            immigrant_modes=tuple(str(x) for x in value.get("immigrant_modes", ())),
            population_cache_stats=dict(value.get("population_cache_stats", {})),
            clone_cache_stats=dict(value.get("clone_cache_stats", {})),
        )


def genes_tuple(chromosome: Chromosome) -> tuple[int, ...]:
    return chromosome.genes

