from __future__ import annotations

import numpy as np
import pytest

from encoding.chromosome import Chromosome, ChromosomeValidationError
from encoding.layout import (
    grid_to_matlab_vector,
    matlab_population_matrix_to_python,
    matlab_vector_to_grid,
    python_population_to_matlab_matrix,
)


def test_chromosome_rejects_invalid_length() -> None:
    with pytest.raises(ChromosomeValidationError, match="length"):
        Chromosome.from_iterable([0] * 179)


@pytest.mark.parametrize("bad", [-1, 3, 9])
def test_chromosome_rejects_gene_outside_trits(bad: int) -> None:
    values = [0] * 180
    values[17] = bad
    with pytest.raises(ChromosomeValidationError, match="genes must be"):
        Chromosome.from_iterable(values)


def test_chromosome_rejects_fractional_gene() -> None:
    values = [0.0] * 180
    values[0] = 1.5
    with pytest.raises(ChromosomeValidationError, match="integers"):
        Chromosome.from_iterable(values)


def test_matlab_layout_round_trip_and_known_indices() -> None:
    chromosome = Chromosome.from_iterable(np.arange(180) % 3)
    grid = matlab_vector_to_grid(chromosome)
    assert grid.shape == (18, 10)
    for radial in range(18):
        for angular in range(10):
            assert grid[radial, angular] == chromosome.genes[radial * 10 + angular]
    assert grid_to_matlab_vector(grid) == chromosome


def test_matlab_population_row_column_and_python_round_trip() -> None:
    first = np.arange(180) % 3
    second = (np.arange(180) + 1) % 3
    matlab_rows = np.stack((first, second))
    python_rows = matlab_population_matrix_to_python(matlab_rows)
    assert python_rows.shape == (2, 180)
    assert np.array_equal(python_population_to_matlab_matrix(python_rows), matlab_rows)
    assert np.array_equal(
        matlab_population_matrix_to_python(first.reshape(180, 1))[0], first
    )


def test_matlab_population_rejects_ambiguous_transposed_multirow_shape() -> None:
    with pytest.raises(ValueError, match="N x 180"):
        matlab_population_matrix_to_python(np.zeros((180, 2), dtype=int))


def test_hashes_are_stable_and_matlab_key_is_exact() -> None:
    chromosome = Chromosome.from_iterable(([0, 1, 2] * 60))
    assert chromosome.matlab_cache_key() == "012" * 60
    assert len(chromosome.sha256()) == 64
    assert chromosome.sha256() == Chromosome.from_iterable(chromosome.genes).sha256()
