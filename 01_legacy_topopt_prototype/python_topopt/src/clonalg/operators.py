from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from encoding.chromosome import CHROMOSOME_LENGTH, Chromosome, ensure_chromosome

from .config import ClonalgConfig, matlab_round_nonnegative


@dataclass(frozen=True, slots=True)
class MutationResult:
    chromosome: Chromosome
    changed_indices: tuple[int, ...]


def mutate_trit(
    chromosome: Chromosome | Sequence[int],
    probability: float,
    rng: np.random.Generator,
) -> MutationResult:
    if probability < 0 or probability > 1:
        raise ValueError("mutation probability must be in [0,1]")
    genes = ensure_chromosome(chromosome).to_numpy(dtype=np.int16)
    # MATLAB source uses ``rand(...) <= pm`` (clonalg_N_float.m:362).
    mask = rng.random(CHROMOSOME_LENGTH) <= probability
    indices = np.flatnonzero(mask)
    if indices.size:
        delta = rng.integers(1, 3, size=indices.size)
        genes[indices] = (genes[indices] + delta) % 3
    return MutationResult(
        chromosome=Chromosome.from_iterable(genes),
        changed_indices=tuple(int(index) for index in indices),
    )


def initialize_population(
    seed: Chromosome | Sequence[int],
    config: ClonalgConfig,
    rng: np.random.Generator,
) -> list[Chromosome]:
    seed_chromosome = ensure_chromosome(seed)
    random_population = rng.integers(
        0, 3, size=(config.population_size, CHROMOSOME_LENGTH)
    )
    population = [Chromosome.from_iterable(row) for row in random_population]
    population[0] = seed_chromosome

    nonseed = config.population_size - 1
    near_count = matlab_round_nonnegative(config.near_fraction_of_nonseed * nonseed)
    middle_count = matlab_round_nonnegative(config.middle_fraction_of_nonseed * nonseed)
    index = 1
    for _ in range(near_count):
        population[index] = mutate_trit(
            seed_chromosome, config.near_mutation_probability, rng
        ).chromosome
        index += 1
    for _ in range(middle_count):
        population[index] = mutate_trit(
            seed_chromosome, config.middle_mutation_probability, rng
        ).chromosome
        index += 1
    if index < config.population_size:
        population[index] = mutate_trit(
            seed_chromosome, config.far_mutation_probability, rng
        ).chromosome
    return population


def stable_fitness_order(fitness: Sequence[float]) -> np.ndarray:
    values = np.asarray(fitness, dtype=float)
    if values.ndim != 1:
        raise ValueError("fitness must be one-dimensional")
    return np.argsort(values, kind="stable")


def select_parent_or_best_clone(
    sorted_population: Sequence[Chromosome],
    sorted_objectives: Sequence[float],
    clones: Sequence[Chromosome],
    clone_objectives: Sequence[float],
    counts: Sequence[int],
) -> list[Chromosome]:
    """Select a clone on equality, matching MATLAB's ``<=`` condition."""

    parent_values = np.asarray(sorted_objectives, dtype=float)
    clone_values = np.asarray(clone_objectives, dtype=float)
    new_population: list[Chromosome] = []
    start = 0
    for parent_rank, count in enumerate(counts):
        stop = start + int(count)
        if stop <= start:
            raise ValueError("every selected parent must have at least one clone")
        local = int(np.argmin(clone_values[start:stop]))
        clone_index = start + local
        if clone_values[clone_index] <= parent_values[parent_rank]:
            new_population.append(clones[clone_index])
        else:
            new_population.append(sorted_population[parent_rank])
        start = stop
    if start != len(clones):
        raise ValueError("clone counts do not cover the clone array")
    new_population.extend(sorted_population[len(counts) :])
    return new_population


def apply_random_immigrants(
    population: list[Chromosome],
    *,
    global_best: Chromosome,
    config: ClonalgConfig,
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Replace historical non-elite positions and then force the elite into row 0."""

    count = max(
        1,
        matlab_round_nonnegative(
            config.random_immigrant_fraction * config.population_size
        ),
    )
    candidates = np.arange(1, config.population_size)
    positions = rng.choice(candidates, size=min(count, len(candidates)), replace=False)
    modes: list[str] = []
    for position_value in positions:
        position = int(position_value)
        if rng.random() < config.immigrant_best_branch_probability:
            population[position] = mutate_trit(
                global_best,
                config.immigrant_best_mutation_probability,
                rng,
            ).chromosome
            modes.append("BEST_MUTATION")
        else:
            population[position] = Chromosome.from_iterable(
                rng.integers(0, 3, size=CHROMOSOME_LENGTH)
            )
            modes.append("FULL_RANDOM")
    enforce_elite(population, global_best)
    return tuple(int(value) for value in positions), tuple(modes)


def enforce_elite(population: list[Chromosome], global_best: Chromosome) -> None:
    if not population:
        raise ValueError("population cannot be empty")
    population[0] = global_best
