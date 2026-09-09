from __future__ import annotations

import math

import pytest

from constraints.connectivity import historical_precheck
from objectives.torque import (
    ObjectiveConfig,
    average_torque,
    average_torque_shortfall_penalty,
    evaluate_historical_objective,
    torque_ripple_ratio,
)


def test_average_and_ripple_hand_calculation() -> None:
    torques = [1.0, 2.0, 3.0]
    assert average_torque(torques) == 2.0
    assert torque_ripple_ratio(torques) == 1.0


def test_low_average_torque_penalty() -> None:
    penalty = average_torque_shortfall_penalty(
        0.6, minimum_average_torque_nm=0.8, penalty_coefficient=10.0
    )
    assert penalty == pytest.approx(2.5)
    assert average_torque_shortfall_penalty(0.8) == 0.0


def test_complete_objective_manual_value(valid_copper_chromosome) -> None:
    precheck = historical_precheck(valid_copper_chromosome)
    assert not precheck.rejected
    torques = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    result = evaluate_historical_objective(torques, precheck=precheck)
    mean = 0.75
    ripple = 0.5 / mean
    shortfall = 10.0 * (0.8 - mean) / 0.8
    p_fe = 0.10 / (1.0 + math.exp(25.0))
    p_cu = 0.05 / (1.0 + math.exp(15.0))
    assert result.average_torque_nm == pytest.approx(mean)
    assert result.torque_ripple_ratio == pytest.approx(ripple)
    assert result.average_torque_penalty == pytest.approx(shortfall)
    assert result.objective == pytest.approx(ripple + shortfall + p_fe + p_cu)


def test_no_copper_is_rejected_after_connectivity() -> None:
    from encoding.chromosome import Chromosome

    chromosome = Chromosome.from_iterable([1] * 180)
    precheck = historical_precheck(chromosome)
    assert not precheck.rejected
    result = evaluate_historical_objective(None, precheck=precheck)
    assert result.status == "NO_COPPER_REJECTED"
    assert result.objective == ObjectiveConfig().invalid_base_objective

