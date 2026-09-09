from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constraints.connectivity import historical_precheck
from encoding.chromosome import Chromosome
from encoding.layout import matlab_vector_to_grid
from objectives.torque import ObjectiveConfig, evaluate_historical_objective

from .base import EvaluationResult


@dataclass(slots=True)
class DeterministicMockEvaluator:
    """Constraint-aware deterministic stand-in; it never calls FEMM."""

    objective_config: ObjectiveConfig = ObjectiveConfig()
    calls: int = 0

    def evaluate(self, chromosome: Chromosome) -> EvaluationResult:
        self.calls += 1
        precheck = historical_precheck(chromosome)
        if precheck.rejected or not precheck.has_any_copper:
            result = evaluate_historical_objective(
                None,
                precheck=precheck,
                config=self.objective_config,
            )
            return EvaluationResult(
                objective=result.objective,
                status=result.status,
                average_torque_nm=result.average_torque_nm,
                torque_ripple_ratio=result.torque_ripple_ratio,
                rejection_reasons=result.rejection_reasons,
                metadata={"mock": True},
            )

        grid = matlab_vector_to_grid(chromosome)
        n_iron = int(np.count_nonzero(grid == 1))
        n_copper = int(np.count_nonzero(grid == 2))
        transitions = int(
            np.count_nonzero(grid[1:, :] != grid[:-1, :])
            + np.count_nonzero(grid[:, 1:] != grid[:, :-1])
        )
        mean_torque = 0.45 + 0.0045 * n_iron + 0.004 * n_copper
        amplitude = 0.01 + 0.00025 * transitions
        shape = np.asarray((-1.0, -0.4, 0.2, 1.0, 0.4, -0.2), dtype=float)
        torque_values = tuple(float(x) for x in mean_torque + amplitude * shape)
        result = evaluate_historical_objective(
            torque_values,
            precheck=precheck,
            config=self.objective_config,
        )
        return EvaluationResult(
            objective=result.objective,
            status=result.status,
            average_torque_nm=result.average_torque_nm,
            torque_ripple_ratio=result.torque_ripple_ratio,
            torque_values_nm=torque_values,
            rejection_reasons=result.rejection_reasons,
            metadata={
                "mock": True,
                "iron_cells": n_iron,
                "copper_cells": n_copper,
                "material_transitions": transitions,
            },
        )

