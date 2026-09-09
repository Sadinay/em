from __future__ import annotations

import numpy as np

from clonalg.operators import stable_fitness_order
from constraints.connectivity import historical_precheck
from data_io.matlab import load_best_result, load_seed, load_trace_all
from topology.historical_inset import decode_historical_inset

from conftest import WORKSPACE_ROOT


TRACE = WORKSPACE_ROOT / "FP" / "IA_test_Float" / "trace_all_20260306_174158.mat"
BEST = WORKSPACE_ROOT / "FP" / "IA_test_Float" / "best_result_20260306_174158.mat"
SEED = WORKSPACE_ROOT / "FP" / "CLONALG" / "seed_bits_fine.mat"


def test_historical_seed_and_best_shapes_and_values() -> None:
    seed = load_seed(SEED)
    assert len(seed.genes) == 180
    assert set(seed.genes) <= {0, 1, 2}

    best = load_best_result(BEST)
    assert best.objective == 0.09249546785567309
    assert best.generation_index_matlab == 1
    assert best.chromosome.material_counts() == {0: 36, 1: 94, 2: 50}
    assert "T_avg" not in best.available_variables
    assert "T_ripple" not in best.available_variables


def test_v73_trace_dimensions_sorting_and_elite_format() -> None:
    trace = load_trace_all(TRACE)
    assert trace.populations.shape == (100, 10, 180)
    assert trace.objectives.shape == (100, 10)
    assert trace.population_size == 10
    assert trace.generations == 100
    assert trace.beta == 0.4
    assert trace.random_immigrant_fraction == 0.2
    for generation in range(trace.generations):
        assert np.array_equal(
            stable_fitness_order(trace.objectives[generation]),
            trace.sorted_indices[generation],
        )
    assert np.all(trace.populations[:, 0, :] == trace.populations[0, 0, :])


def test_historical_precheck_regression_documents_source_output_drift() -> None:
    trace = load_trace_all(TRACE)
    computed_iron = np.zeros((100, 10), dtype=int)
    computed_copper = np.zeros((100, 10), dtype=int)
    computed_objective = np.zeros((100, 10), dtype=float)
    for generation in range(100):
        for member in range(10):
            result = historical_precheck(trace.populations[generation, member])
            computed_iron[generation, member] = result.iron.n_floating
            computed_copper[generation, member] = result.copper.n_floating
            if result.rejected:
                computed_objective[generation, member] = result.objective_if_rejected
    # The copper detector in the checked-in MATLAB source is reproduced exactly.
    assert np.array_equal(computed_copper, trace.small_copper_cells)

    # The present detect_floating_iron.m reproduces 901/1000 saved positions.
    # The other 99 positions (55 unique chromosomes) have deterministic saved
    # counts that differ from the present source.  Keep this assertion explicit:
    # it turns an unexplained historical version drift into a visible contract.
    iron_matches = computed_iron == trace.floating_iron_cells
    assert int(np.count_nonzero(iron_matches)) == 901
    mismatch_chromosomes = {
        tuple(trace.populations[generation, member])
        for generation, member in np.argwhere(~iron_matches)
    }
    assert len(mismatch_chromosomes) == 55

    rejected = (computed_iron > 0) | (computed_copper > 0)
    assert int(np.count_nonzero(rejected)) == 900
    assert np.array_equal(rejected, trace.objectives >= 1_000_000)

    # The historical objective values are exactly explained by the counters
    # stored in the MAT trace, even where the current iron source has drifted.
    stored_counter_objective = (
        1_000_000.0
        + 50.0 * trace.floating_iron_cells
        + 50.0 * trace.small_copper_cells
    )
    historical_rejected = trace.objectives >= 1_000_000
    assert np.array_equal(
        stored_counter_objective[historical_rejected],
        trace.objectives[historical_rejected],
    )
    assert int(
        np.count_nonzero(
            computed_objective[historical_rejected]
            == trace.objectives[historical_rejected]
        )
    ) == 801


def test_historical_best_inset_layout_material_counts() -> None:
    best = load_best_result(BEST)
    topology = decode_historical_inset(best.chromosome)
    assert int(np.count_nonzero(topology.base_material_grid == 0)) == 36
    assert int(np.count_nonzero(topology.base_material_grid == 1)) == 94
    assert int(np.count_nonzero(topology.base_material_grid == 2)) == 50
    assert sum(cell.material_code == 2 for cell in topology.cells) == 6 * 50
    assert all(
        cell.copper_inset_radial_bounds_mm is not None
        for cell in topology.cells
        if cell.material_code == 2
    )
