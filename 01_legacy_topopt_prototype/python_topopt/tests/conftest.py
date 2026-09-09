from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from encoding.chromosome import Chromosome
from encoding.layout import grid_to_matlab_vector


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def valid_copper_chromosome() -> Chromosome:
    grid = np.ones((18, 10), dtype=np.uint8)
    grid[5:7, 4:6] = 2  # exactly four connected copper cells; iron remains anchored.
    return grid_to_matlab_vector(grid)

