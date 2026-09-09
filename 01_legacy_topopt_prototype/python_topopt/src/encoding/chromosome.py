from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np


CHROMOSOME_LENGTH = 180
ALLOWED_GENES = (0, 1, 2)


class ChromosomeValidationError(ValueError):
    """Raised when a chromosome cannot represent the historical 180-trit design."""


@dataclass(frozen=True, slots=True)
class Chromosome:
    """Immutable 180-trit chromosome used by the historical MATLAB model."""

    genes: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.genes) != CHROMOSOME_LENGTH:
            raise ChromosomeValidationError(
                f"chromosome length {len(self.genes)} != {CHROMOSOME_LENGTH}"
            )
        invalid = sorted({value for value in self.genes if value not in ALLOWED_GENES})
        if invalid:
            raise ChromosomeValidationError(
                f"genes must be in {ALLOWED_GENES}; found {invalid}"
            )

    @classmethod
    def from_iterable(cls, values: Iterable[int]) -> "Chromosome":
        array = np.asarray(list(values))
        if array.ndim != 1:
            raise ChromosomeValidationError(
                f"chromosome must be one-dimensional; got shape {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise ChromosomeValidationError("chromosome genes must be numeric integers")
        if np.any(~np.isfinite(array.astype(float))):
            raise ChromosomeValidationError("chromosome genes must be finite")
        if np.any(array != np.floor(array.astype(float))):
            raise ChromosomeValidationError("chromosome genes must be integers")
        return cls(tuple(int(value) for value in array.tolist()))

    def to_numpy(self, *, dtype: np.dtype | type = np.uint8) -> np.ndarray:
        return np.asarray(self.genes, dtype=dtype).copy()

    def matlab_cache_key(self) -> str:
        """Exact key shape used by MATLAB: 180 literal characters '0'/'1'/'2'."""

        return "".join(str(value) for value in self.genes)

    def sha256(self) -> str:
        """Stable content hash for persistent records and topology identity."""

        return hashlib.sha256(bytes(self.genes)).hexdigest()

    def material_counts(self) -> dict[int, int]:
        array = self.to_numpy()
        return {code: int(np.count_nonzero(array == code)) for code in ALLOWED_GENES}


def ensure_chromosome(value: Chromosome | Iterable[int]) -> Chromosome:
    return value if isinstance(value, Chromosome) else Chromosome.from_iterable(value)

