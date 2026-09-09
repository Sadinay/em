from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from encoding.chromosome import Chromosome


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    objective: float
    status: str
    average_torque_nm: float = 0.0
    torque_ripple_ratio: float = 0.0
    torque_values_nm: tuple[float, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": float(self.objective),
            "status": self.status,
            "average_torque_nm": float(self.average_torque_nm),
            "torque_ripple_ratio": float(self.torque_ripple_ratio),
            "torque_values_nm": list(self.torque_values_nm),
            "rejection_reasons": list(self.rejection_reasons),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationResult":
        return cls(
            objective=float(value["objective"]),
            status=str(value["status"]),
            average_torque_nm=float(value.get("average_torque_nm", 0.0)),
            torque_ripple_ratio=float(value.get("torque_ripple_ratio", 0.0)),
            torque_values_nm=tuple(float(x) for x in value.get("torque_values_nm", ())),
            rejection_reasons=tuple(str(x) for x in value.get("rejection_reasons", ())),
            metadata=dict(value.get("metadata", {})),
        )


class Evaluator(Protocol):
    def evaluate(self, chromosome: Chromosome) -> EvaluationResult:
        ...

