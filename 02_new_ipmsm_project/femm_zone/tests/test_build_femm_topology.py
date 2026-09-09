from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "femm_zone" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_builder():
    path = SCRIPTS / "build_femm_topology.py"
    spec = importlib.util.spec_from_file_location("build_femm_topology", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_builder()


def test_two_bit_decode_and_physical_mapping() -> None:
    pairs = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8), (25, 1))
    codes = builder.decode_bits(pairs.reshape(-1))
    classes = builder.physical_classes(codes)
    assert codes[:4].tolist() == [0, 1, 2, 3]
    assert classes[:4].tolist() == [0, 1, 2, 2]


def test_codes_two_and_three_share_physical_hash() -> None:
    classes_2 = builder.physical_classes(np.full(100, 2, dtype=np.uint8))
    classes_3 = builder.physical_classes(np.full(100, 3, dtype=np.uint8))
    assert builder.topology_hash(classes_2) == builder.topology_hash(classes_3)


def test_generated_model_matches_all_400_labels(tmp_path: Path) -> None:
    bits, provenance = builder.gene_from_mat(
        ROOT / "data_zone" / "raw" / "workspace_600.mat",
        state_index=600,
        population_row=None,
        gene_source="before_correction",
    )
    model = builder.build_topology(
        bits,
        ROOT / "femm_zone" / "models" / "IPMSM.fem",
        ROOT / "data_zone" / "raw" / "workspace_600.mat",
        tmp_path,
        provenance,
    )
    assert model.exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation"]["mapped_cells"] == 100
    assert manifest["validation"]["mapped_design_labels"] == 400
    assert manifest["validation"]["all_expected_materials_match"] is True
    assert manifest["validation"]["physical_class_counts"] == {
        "air": 11,
        "permanent_magnet": 27,
        "iron": 62,
    }
    assert manifest["validation"]["pm_label_count"] == 27 * 4
    assert manifest["validation"]["maximum_pm_magnetization_direction_error_deg"] < 1e-9
    assert manifest["validation"]["all_pm_magnetization_directions_match_template_rule"] is True


def test_template_pm_direction_rule() -> None:
    assert builder.expected_pm_magnetization_deg(1.0, 0.0) == 180.0
    assert builder.expected_pm_magnetization_deg(1.0, 1.0) == 45.0
    assert builder.expected_pm_magnetization_deg(0.0, 1.0) == 90.0
    assert builder.angular_distance_deg(-136.125, 223.875) < 1e-12
