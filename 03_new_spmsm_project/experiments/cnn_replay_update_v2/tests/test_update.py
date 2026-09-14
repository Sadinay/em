"""Offline checks for replay identity and the predeclared selection boundary."""
import copy
from contextlib import nullcontext
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
    old = update.cycle_plan(40000, 56, 2000, update.SEED+1)
    new = update.cycle_plan(1400, 8, 2000, update.SEED+2)
    assert sorted(old.ravel()[:40000]) == list(range(40000))
    assert sorted(old.ravel()[40000:80000]) == list(range(40000))
    assert sorted(new.ravel()[:1400]) == list(range(1400))
    microbatches = list(update.ReplaySampler(old, new, 837, 839))
    for micro in microbatches:
        assert sum(i < 40000 for i in micro) == 7
        assert sum(i >= 40000 for i in micro) == 1
    whole = list(update.ReplaySampler(old, new, 0, 839))
    assert microbatches == whole[837*8:]


def test_iterator_restart_does_not_consume_dropout_rng():
    data = update.array_dataset(np.zeros((24,120), np.uint8), np.zeros((24,2), np.float32))
    torch.manual_seed(101)
    before = torch.get_rng_state().clone()
    next(iter(DataLoader(data, batch_size=8, generator=torch.Generator().manual_seed(102))))
    assert torch.equal(before, torch.get_rng_state())


def test_accumulated_gradient_matches_full_batch_and_weighted_loss(monkeypatch):
    torch.manual_seed(87)
    model = torch.nn.Linear(3, 2)
    full_model = copy.deepcopy(model)
    x, y = torch.randn(64, 3), torch.randn(64, 2)
    mean, std = torch.tensor([3.1, .4]), torch.tensor([.357, .191])
    batches = [(torch.arange(i, i+8), x[i:i+8], y[i:i+8]) for i in range(0, 64, 8)]
    monkeypatch.setattr(update.tr, 'make_inputs', lambda bits, spec, renderer: bits)
    monkeypatch.setattr(torch.amp, 'autocast', lambda *args, **kwargs: nullcontext())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    reference_optimizer = torch.optim.SGD(full_model.parameters(), lr=1e-4)
    value = update.perform_update(model, None, None, mean, std, torch.device('cpu'), optimizer,
                                  torch.amp.GradScaler('cuda', enabled=False), batches)
    loss = torch.nn.functional.mse_loss(full_model(x), (y-mean)/std)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(full_model.parameters(), 100.0)
    reference_optimizer.step()
    for p, q in zip(model.parameters(), full_model.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)
    assert value['mse'] == pytest.approx(loss.item(), rel=1e-6)
    assert value['mse'] == pytest.approx(.875*value['old_mse']+.125*value['new_mse'], rel=1e-6)


def test_acceptance_requires_new_improvement_and_both_old_limits():
    base = {'old_validation': {'metrics': {'tavg': {'mae': .01}, 'delta_t': {'mae': .02}}},
            'dev_common': {'standardized_mse': .8}}
    candidate = copy.deepcopy(base)
    assert not update.improvement_eligible(candidate, base)
    candidate['dev_common']['standardized_mse'] = .1
    assert update.improvement_eligible(candidate, base)
    candidate['old_validation']['metrics']['delta_t']['mae'] = .021001
    assert not update.improvement_eligible(candidate, base)


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
