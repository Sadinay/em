from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from femm_zone import femm_config  # noqa: E402
from femm_zone.scripts import spmsm_mapping  # noqa: E402
from femm_zone.scripts.safe_boundary_merge import merge_equivalent_design_cells  # noqa: E402


def inputs():
    template = femm_config.TEMPLATE_FILE.read_text(encoding="utf-8")
    positions = loadmat(femm_config.MAT_FILE, variable_names=["MaterialPosition"],
                        squeeze_me=True)["MaterialPosition"]
    return template, spmsm_mapping.match_material_positions(template, positions)


def test_all_air_merges_each_copy_without_touching_fixed_regions():
    template, rows = inputs()
    bits = np.zeros(120, dtype=np.uint8)
    generated = spmsm_mapping.replace_cell_materials(template, bits)
    merged, audit = merge_equivalent_design_cells(generated, bits, rows)
    assert audit["removed_segments"] == 19 * 6 * 4
    assert audit["removed_arc_segments"] == 20 * 5 * 4
    assert audit["remaining_block_labels"] == 12 + 4
    assert len(spmsm_mapping.parse_labels(merged)) == 16


def test_pm_direction_boundaries_are_preserved():
    template, rows = inputs()
    bits = np.ones(120, dtype=np.uint8)
    generated = spmsm_mapping.replace_cell_materials(template, bits)
    _, audit = merge_equivalent_design_cells(generated, bits, rows)
    # PM direction is constant radially, but changes between angular columns.
    assert audit["removed_arc_segments"] == 20 * 5 * 4
    assert audit["removed_segments"] == 0
    assert audit["skipped_pm_boundaries_with_different_direction"] == 19 * 6 * 4
    assert audit["remaining_block_labels"] == 12 + 20 * 4


def test_mixed_connected_gene_keeps_model_sections_well_formed():
    template, rows = inputs()
    bits = np.zeros(120, dtype=np.uint8)
    bits[:12] = 1
    generated = spmsm_mapping.replace_cell_materials(template, bits)
    merged, audit = merge_equivalent_design_cells(generated, bits, rows)
    assert audit["removed_segments"] > 0
    assert audit["removed_arc_segments"] > 0
    for section, expected in (("NumPoints", audit["remaining_nodes"]),
                              ("NumSegments", audit["remaining_segments"]),
                              ("NumArcSegments", audit["remaining_arc_segments"]),
                              ("NumBlockLabels", audit["remaining_block_labels"])):
        import re
        assert int(re.search(rf"\[{section}\]\s*=\s*(\d+)", merged).group(1)) == expected
