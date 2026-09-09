from __future__ import annotations

import math

import numpy as np
import pytest

from cnn_zone.semantic90.src.training_v2 import (
    EffectiveBatchSampler,
    V2_TRAINING_SPECS,
    complete_metrics,
)


def test_effective_batch_sampler_uses_every_sample_once() -> None:
    sampler = EffectiveBatchSampler(11_700, physical_batch_size=24, effective_batch_size=64, seed=7)
    microbatches = list(sampler)
    flat = np.concatenate([np.asarray(batch, dtype=np.int64) for batch in microbatches])
    assert len(flat) == 11_700
    assert len(np.unique(flat)) == 11_700
    assert flat.min() == 0 and flat.max() == 11_699
    assert all(0 < len(batch) <= 24 for batch in microbatches)
    assert sampler.optimizer_steps_per_epoch == math.ceil(11_700 / 64) == 183


def test_effective_groups_are_exactly_64_except_final_group() -> None:
    sampler = EffectiveBatchSampler(11_700, physical_batch_size=24, effective_batch_size=64, seed=9)
    group_totals = []
    running = 0
    for batch in sampler:
        running += len(batch)
        if running == 64:
            group_totals.append(running)
            running = 0
        elif len(group_totals) == 182 and running == 52:
            group_totals.append(running)
            running = 0
    assert group_totals == [64] * 182 + [52]


def test_all_training_specs_use_effective_batch_64() -> None:
    assert len(V2_TRAINING_SPECS) == 6
    assert all(spec.effective_batch_size == 64 for spec in V2_TRAINING_SPECS.values())


def test_complete_metrics_and_bands_include_counts() -> None:
    actual = np.asarray([[0.2, 0.3], [0.7, 0.6], [1.2, 0.9], [3.1, 2.1]], dtype=np.float64)
    predicted = actual + np.asarray([[0.1, -0.1], [-0.1, 0.1], [0.0, 0.2], [0.2, -0.2]])
    metrics = complete_metrics(actual, predicted)
    assert set(metrics["overall"]) == {"tavg", "delta_t"}
    assert metrics["overall"]["tavg"]["mae"] == pytest.approx(0.1)
    assert sum(item["count"] for item in metrics["bands"]["tavg"]) == 4
    assert sum(item["count"] for item in metrics["bands"]["delta_t"]) == 4
    assert all("mae" in item and "bias" in item for item in metrics["bands"]["tavg"])
