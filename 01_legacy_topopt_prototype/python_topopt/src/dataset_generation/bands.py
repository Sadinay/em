from __future__ import annotations

import math


BAND_RANGES: tuple[tuple[str, float, float | None], ...] = (
    ("B1", 0.0, 0.5),
    ("B2", 0.5, 0.6),
    ("B3", 0.6, 0.7),
    ("B4", 0.7, 0.8),
    ("B5", 0.8, 0.9),
    ("B6", 0.9, 1.0),
    ("B7", 1.0, None),
)


def classify_torque_ratio(value: float) -> str:
    ratio = float(value)
    if not math.isfinite(ratio):
        raise ValueError("torque ratio must be finite")
    if ratio < 0:
        return "negative_torque"
    for name, lower, upper in BAND_RANGES:
        if ratio >= lower and (upper is None or ratio < upper):
            return name
    raise AssertionError("finite non-negative ratio did not match a band")
