"""Offline checks for replay identity and the predeclared selection boundary."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

spec = importlib.util.spec_from_file_location("replay_update", Path(__file__).parents[1] / "update.py")
update = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update)


def test_replay_cycles_cover_every_old_sample_before_repeating():
    old = update.cycle_plan(40000, 48, 2000, update.SEED+1)
    new = update.cycle_plan(1400, 16, 2000, update.SEED+2)
    assert sorted(old.ravel()[:40000]) == list(range(40000))
    assert sorted(old.ravel()[40000:80000]) == list(range(40000))
    assert sorted(new.ravel()[:1400]) == list(range(1400))
    microbatches = list(update.ReplaySampler(old, new, 837, 839))
    for micro in microbatches:
        assert sum(i < 40000 for i in micro) == 6
        assert sum(i >= 40000 for i in micro) == 2
    whole = list(update.ReplaySampler(old, new, 0, 839))
    assert microbatches == whole[837*8:]


def test_iterator_restart_does_not_consume_dropout_rng():
    data = update.array_dataset(np.zeros((24,120), np.uint8), np.zeros((24,2), np.float32))
    torch.manual_seed(101)
    before = torch.get_rng_state().clone()
    next(iter(DataLoader(data, batch_size=8, generator=torch.Generator().manual_seed(102))))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("bad_target", update.TARGETS)
def test_constraint_requires_both_old_targets(bad_target):
    baseline = {"old_validation": {"metrics": {"tavg": {"mae": .01}, "delta_t": {"mae": .02}}}}
    candidate = copy.deepcopy(baseline)
    for target in update.TARGETS:
        candidate["old_validation"]["metrics"][target]["mae"] *= 1.05
    assert update.admissible(candidate, baseline)
    candidate["old_validation"]["metrics"][bad_target]["mae"] += 1e-8
    assert not update.admissible(candidate, baseline)


@pytest.mark.parametrize("role", ("test", "test_common", "old_test", "sealed_test"))
def test_test_roles_rejected_before_loading(role):
    with pytest.raises(RuntimeError, match="Sealed"):
        update.allowed_dataset({}, role)
