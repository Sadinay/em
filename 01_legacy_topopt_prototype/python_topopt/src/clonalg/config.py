from __future__ import annotations

from dataclasses import asdict, dataclass
import math


def matlab_round_nonnegative(value: float) -> int:
    if value < 0:
        raise ValueError("this compatibility helper accepts non-negative values only")
    return int(math.floor(value + 0.5))


@dataclass(frozen=True, slots=True)
class ClonalgConfig:
    population_size: int = 10
    generations: int = 100
    beta: float = 0.4
    random_immigrant_fraction: float = 0.2
    near_fraction_of_nonseed: float = 0.6
    middle_fraction_of_nonseed: float = 0.2
    near_mutation_probability: float = 0.05
    middle_mutation_probability: float = 0.12
    far_mutation_probability: float = 0.25
    hypermutation_min_probability: float = 0.03
    hypermutation_max_probability: float = 0.20
    hypermutation_rank_exponent: float = 2.0
    immigrant_best_mutation_probability: float = 0.25
    immigrant_best_branch_probability: float = 0.5
    elite_count: int = 1
    random_seed: int = 20260806

    def __post_init__(self) -> None:
        if self.population_size < 3:
            raise ValueError("historical CLONALG needs a population of at least 3")
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if self.elite_count != 1:
            raise ValueError("historical compatibility mode has exactly one elite")
        probabilities = (
            self.random_immigrant_fraction,
            self.near_fraction_of_nonseed,
            self.middle_fraction_of_nonseed,
            self.near_mutation_probability,
            self.middle_mutation_probability,
            self.far_mutation_probability,
            self.hypermutation_min_probability,
            self.hypermutation_max_probability,
            self.immigrant_best_mutation_probability,
            self.immigrant_best_branch_probability,
        )
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("probabilities and fractions must be in [0,1]")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def clone_counts(config: ClonalgConfig) -> tuple[int, ...]:
    selected_count = max(0, config.population_size - 2)
    if selected_count == 0:
        return ()
    maximum = config.beta * config.population_size
    minimum = 1.0
    denominator = max(1, selected_count - 1)
    values = []
    for zero_based_rank in range(selected_count):
        raw = maximum - (maximum - minimum) * zero_based_rank / denominator
        values.append(max(1, matlab_round_nonnegative(raw)))
    return tuple(values)


def hypermutation_probabilities(config: ClonalgConfig) -> tuple[float, ...]:
    selected_count = max(0, config.population_size - 2)
    if selected_count == 0:
        return ()
    denominator = max(1, selected_count - 1)
    return tuple(
        config.hypermutation_min_probability
        + (
            config.hypermutation_max_probability
            - config.hypermutation_min_probability
        )
        * (rank / denominator) ** config.hypermutation_rank_exponent
        for rank in range(selected_count)
    )

