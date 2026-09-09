from __future__ import annotations

import numpy as np

from .chromosome import CHROMOSOME_LENGTH, Chromosome, ensure_chromosome


RADIAL_CELLS = 18
ANGULAR_CELLS = 10


def matlab_population_matrix_to_python(matrix: np.ndarray) -> np.ndarray:
    """Validate a MATLAB logical ``N x 180`` chromosome matrix.

    A single MATLAB row (``1 x 180``), column (``180 x 1``), or squeezed
    vector is normalized to ``1 x 180``.  No transpose is guessed for a
    multi-individual ``180 x N`` array; HDF5 MAT dimension reversal belongs
    in the MAT loader where the file format is known.
    """

    array = np.asarray(matrix)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim == 2 and array.shape == (CHROMOSOME_LENGTH, 1):
        array = array.T
    if array.ndim != 2 or array.shape[1] != CHROMOSOME_LENGTH:
        raise ValueError(
            f"MATLAB population shape {array.shape} must be N x {CHROMOSOME_LENGTH}"
        )
    rows = [Chromosome.from_iterable(row).to_numpy() for row in array]
    return np.stack(rows, axis=0).astype(np.uint8, copy=False)


def python_population_to_matlab_matrix(population: np.ndarray) -> np.ndarray:
    """Return a validated ``N x 180`` array suitable for ``scipy.io.savemat``."""

    return matlab_population_matrix_to_python(population).copy()


def matlab_vector_to_grid(
    chromosome: Chromosome | np.ndarray | list[int] | tuple[int, ...],
) -> np.ndarray:
    """Reproduce MATLAB ``reshape(bits,[nt,nr]).'`` as an 18x10 grid."""

    genes = ensure_chromosome(chromosome).to_numpy()
    return genes.reshape((ANGULAR_CELLS, RADIAL_CELLS), order="F").T.copy()


def grid_to_matlab_vector(grid: np.ndarray) -> Chromosome:
    """Inverse of :func:`matlab_vector_to_grid` with MATLAB column ordering."""

    array = np.asarray(grid)
    expected = (RADIAL_CELLS, ANGULAR_CELLS)
    if array.shape != expected:
        raise ValueError(f"grid shape {array.shape} != {expected}")
    vector = array.T.reshape(CHROMOSOME_LENGTH, order="F")
    return Chromosome.from_iterable(vector)
