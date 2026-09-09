from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_smoke_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_smoke_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_requested_average_definitions() -> None:
    torques = np.asarray([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
    values = MODULE.average_candidates(torques)
    assert values["mean_6"] == 7.5
    assert values["mean_5_excluding_15_deg"] == 6.0
    assert values["trapz_0_to_15_deg"] == 7.5


def test_requested_delta_definitions() -> None:
    values = MODULE.delta_candidates(np.asarray([1.0, 3.0]), average=2.0)
    assert values["absolute_peak_to_peak"] == 2.0
    assert values["peak_to_peak_over_average"] == 1.0
    assert values["percent_peak_to_peak_over_average"] == 100.0
    assert values["half_peak_to_peak_over_average"] == 0.5
