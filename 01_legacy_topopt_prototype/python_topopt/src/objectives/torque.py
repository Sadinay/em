from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from constraints.connectivity import PrecheckResult


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    minimum_average_torque_nm: float = 0.8
    penalty_coefficient: float = 10.0
    ripple_denominator_floor_nm: float = 1.0e-6
    invalid_base_objective: float = 1_000_000.0
    floating_iron_sigmoid_k: float = 50.0
    floating_iron_sigmoid_weight: float = 0.10
    floating_copper_sigmoid_k: float = 30.0
    floating_copper_sigmoid_weight: float = 0.05


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    objective: float
    average_torque_nm: float
    torque_ripple_ratio: float
    average_torque_penalty: float
    floating_iron_penalty: float
    floating_copper_penalty: float
    status: str
    rejection_reasons: tuple[str, ...] = ()


def _validated_torque_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("torque values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("torque values must all be finite")
    return array


def average_torque(values: Sequence[float]) -> float:
    return float(np.mean(_validated_torque_array(values)))


def torque_ripple_ratio(
    values: Sequence[float],
    *,
    denominator_floor_nm: float = 1.0e-6,
) -> float:
    array = _validated_torque_array(values)
    mean = float(np.mean(array))
    denominator = max(abs(mean), float(denominator_floor_nm))
    return float((np.max(array) - np.min(array)) / denominator)


def average_torque_shortfall_penalty(
    average_torque_nm: float,
    *,
    minimum_average_torque_nm: float = 0.8,
    penalty_coefficient: float = 10.0,
) -> float:
    if average_torque_nm < minimum_average_torque_nm:
        return float(
            penalty_coefficient
            * (minimum_average_torque_nm - average_torque_nm)
            / minimum_average_torque_nm
        )
    return 0.0


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def evaluate_historical_objective(
    torque_values_nm: Sequence[float] | None,
    *,
    precheck: PrecheckResult,
    config: ObjectiveConfig = ObjectiveConfig(),
) -> ObjectiveResult:
    """Apply hard rejects first, then the exact historical scalar formulas."""

    if precheck.rejected:
        assert precheck.objective_if_rejected is not None
        return ObjectiveResult(
            objective=float(precheck.objective_if_rejected),
            average_torque_nm=0.0,
            torque_ripple_ratio=0.0,
            average_torque_penalty=0.0,
            floating_iron_penalty=0.0,
            floating_copper_penalty=0.0,
            status="PRECHECK_REJECTED",
            rejection_reasons=precheck.reasons,
        )
    if not precheck.has_any_copper:
        return ObjectiveResult(
            objective=float(config.invalid_base_objective),
            average_torque_nm=0.0,
            torque_ripple_ratio=0.0,
            average_torque_penalty=0.0,
            floating_iron_penalty=0.0,
            floating_copper_penalty=0.0,
            status="NO_COPPER_REJECTED",
            rejection_reasons=("NO_COPPER",),
        )
    if torque_values_nm is None:
        raise ValueError("torque values are required for a topology that passes precheck")

    mean = average_torque(torque_values_nm)
    ripple = torque_ripple_ratio(
        torque_values_nm,
        denominator_floor_nm=config.ripple_denominator_floor_nm,
    )
    torque_penalty = average_torque_shortfall_penalty(
        mean,
        minimum_average_torque_nm=config.minimum_average_torque_nm,
        penalty_coefficient=config.penalty_coefficient,
    )
    iron_penalty = config.floating_iron_sigmoid_weight * _sigmoid(
        config.floating_iron_sigmoid_k * (precheck.iron.n_floating - 0.5)
    )
    copper_penalty = config.floating_copper_sigmoid_weight * _sigmoid(
        config.floating_copper_sigmoid_k * (precheck.copper.n_floating - 0.5)
    )
    objective = ripple + torque_penalty + iron_penalty + copper_penalty
    return ObjectiveResult(
        objective=float(objective),
        average_torque_nm=mean,
        torque_ripple_ratio=ripple,
        average_torque_penalty=torque_penalty,
        floating_iron_penalty=float(iron_penalty),
        floating_copper_penalty=float(copper_penalty),
        status="SUCCEEDED",
    )

