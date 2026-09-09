from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "femm_zone" / "scripts"))

from spmsm_mapping import (  # noqa: E402
    GENE_COUNT,
    match_material_positions,
    material_blocks,
    material_class,
    replace_cell_materials,
)


MAT = ROOT / "data_zone" / "raw" / "workspace_200.mat"
FEM = ROOT / "femm_zone" / "models" / "SPMSM_discrete.fem"


def test_material_positions_match_480_unique_fem_labels() -> None:
    positions = np.asarray(loadmat(MAT, variable_names=["MaterialPosition"])["MaterialPosition"])
    rows = match_material_positions(FEM.read_text(encoding="utf-8"), positions)
    assert len(rows) == 480
    assert len({row["label_index_1based"] for row in rows}) == 480
    assert max(row["coordinate_error_mm"] for row in rows) < 1e-8


def test_gene_order_is_six_radial_cells_per_angle() -> None:
    positions = np.asarray(loadmat(MAT, variable_names=["MaterialPosition"])["MaterialPosition"])
    xy = positions[:, 0:2]
    angle = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))
    radius = np.hypot(xy[:, 0], xy[:, 1])
    for start in range(0, GENE_COUNT, 6):
        assert np.ptp(angle[start : start + 6]) < 1e-10
        assert np.all(np.diff(radius[start : start + 6]) > 0)
    assert np.all(np.diff(angle[::6]) < 0)


def test_one_bit_maps_to_n38_and_zero_bit_maps_to_air() -> None:
    template = FEM.read_text(encoding="utf-8")
    bits = np.zeros(GENE_COUNT, dtype=np.uint8)
    bits[0] = 1
    generated = replace_cell_materials(template, bits)
    by_name = {}
    for block in material_blocks(generated):
        import re

        name = re.search(r'<BlockName>\s*=\s*"([^"]+)"', block).group(1)
        by_name[name] = block
    assert material_class(by_name["c1"]) == "permanent_magnet"
    assert material_class(by_name["c2"]) == "air_or_nonmagnetic"


def test_volume_pm_confirms_final_state_one_bit_semantics() -> None:
    data = loadmat(MAT, variable_names=["population_all", "VolumePM_all"], squeeze_me=True)
    final_bits = np.asarray(data["population_all"], dtype=np.uint8)[:, :, -1]
    final_volume = np.asarray(data["VolumePM_all"], dtype=np.int64)[:, -1]
    assert np.array_equal(final_bits.sum(axis=1), final_volume)
