from __future__ import annotations

import numpy as np

from clonalg.config import ClonalgConfig, clone_counts, hypermutation_probabilities
from clonalg.operators import (
    apply_random_immigrants,
    enforce_elite,
    initialize_population,
    mutate_trit,
    select_parent_or_best_clone,
    stable_fitness_order,
)
from encoding.chromosome import Chromosome


def _chromosome(value: int) -> Chromosome:
    return Chromosome.from_iterable([value] * 180)


def test_clone_counts_and_hypermutation_limits() -> None:
    config = ClonalgConfig()
    assert clone_counts(config) == (4, 4, 3, 3, 2, 2, 1, 1)
    probabilities = hypermutation_probabilities(config)
    assert probabilities[0] == 0.03
    assert probabilities[-1] == 0.20
    assert all(a <= b for a, b in zip(probabilities, probabilities[1:]))


def test_hypermutation_probability_boundaries() -> None:
    original = _chromosome(1)
    unchanged = mutate_trit(original, 0.0, np.random.default_rng(1))
    assert unchanged.chromosome == original
    assert unchanged.changed_indices == ()

    changed = mutate_trit(original, 1.0, np.random.default_rng(1))
    assert len(changed.changed_indices) == 180
    assert all(value in (0, 2) for value in changed.chromosome.genes)


def test_parent_clone_selection_uses_clone_on_tie() -> None:
    parents = [_chromosome(0), _chromosome(1), _chromosome(2)]
    clone_a = Chromosome.from_iterable([1] + [0] * 179)
    clone_b = Chromosome.from_iterable([2] + [0] * 179)
    selected = select_parent_or_best_clone(
        parents,
        [1.0, 2.0, 3.0],
        [clone_a, clone_b],
        [1.0, 1.5],
        [2],
    )
    assert selected[0] == clone_a
    assert selected[1:] == parents[1:]


def test_elite_is_never_lost() -> None:
    population = [_chromosome(0) for _ in range(10)]
    elite = _chromosome(2)
    enforce_elite(population, elite)
    assert population[0] == elite


def test_random_immigrant_count_and_elite() -> None:
    config = ClonalgConfig()
    population = [_chromosome(0) for _ in range(10)]
    elite = _chromosome(1)
    positions, modes = apply_random_immigrants(
        population,
        global_best=elite,
        config=config,
        rng=np.random.default_rng(123),
    )
    assert len(positions) == 2
    assert len(set(positions)) == 2
    assert all(position != 0 for position in positions)
    assert len(modes) == 2
    assert population[0] == elite


def test_initialization_is_reproducible() -> None:
    seed = _chromosome(1)
    config = ClonalgConfig(random_seed=17)
    first = initialize_population(seed, config, np.random.default_rng(17))
    second = initialize_population(seed, config, np.random.default_rng(17))
    assert first == second
    assert first[0] == seed
    assert len(first) == 10


def test_stable_sort_preserves_input_order_for_ties() -> None:
    order = stable_fitness_order([2.0, 1.0, 1.0, 3.0, 1.0])
    assert order.tolist() == [1, 2, 4, 0, 3]

