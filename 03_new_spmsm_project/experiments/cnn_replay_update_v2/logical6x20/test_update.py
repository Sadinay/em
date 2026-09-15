"""Focused checks for both logical replay proportions and frozen model semantics."""
from contextlib import nullcontext
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch


@pytest.fixture(params=['v1','v2'])
def runner(request):
    path=Path(__file__).parents[2]/f'cnn_replay_update_{request.param}'/'logical6x20/update.py'
    spec=importlib.util.spec_from_file_location('logical_'+request.param,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_full_batch_ratio_cycles_and_resumption(runner):
    old=runner.cycle_plan(40000,runner.OLD_PER,runner.MAX_UPDATES,runner.SEED+1)
    new=runner.cycle_plan(1400,runner.NEW_PER,runner.MAX_UPDATES,runner.SEED+2)
    assert new.size==160000
    assert sorted(old.ravel()[:40000])==list(range(40000))
    assert sorted(new.ravel()[:1400])==list(range(1400))
    batch=list(runner.ReplaySampler(old,new,837,839))
    assert len(batch)==2
    for b in batch:
        assert len(b)==64 and sum(i<40000 for i in b)==runner.OLD_PER
    assert batch==list(runner.ReplaySampler(old,new,0,839))[837:]


def test_loss_is_full_batch_mean_and_component_weights(runner,monkeypatch):
    torch.manual_seed(100)
    model=torch.nn.Linear(3,2);reference=copy.deepcopy(model)
    x,y=torch.randn(64,3),torch.randn(64,2)
    mean,std=torch.tensor([3.1,.4]),torch.tensor([.357,.191])
    monkeypatch.setattr(runner.tr,'make_inputs',lambda bits,spec,renderer:bits)
    monkeypatch.setattr(torch.amp,'autocast',lambda *a,**kw:nullcontext())
    opt=torch.optim.SGD(model.parameters(),lr=1e-4);refopt=torch.optim.SGD(reference.parameters(),lr=1e-4)
    result=runner.perform_update(model,None,None,mean,std,torch.device('cpu'),opt,
        torch.amp.GradScaler('cuda',enabled=False),[(torch.arange(64),x,y)])
    loss=torch.nn.functional.mse_loss(reference(x),(y-mean)/std);loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(),100.0);refopt.step()
    for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p,q)
    assert result['mse']==pytest.approx(loss.item(),rel=1e-6)
    assert result['mse']==pytest.approx((runner.OLD_PER*result['old_mse']+runner.NEW_PER*result['new_mse'])/64,rel=1e-6)


def test_selection_and_sealed_role_guard(runner):
    base={'old_validation':{'metrics':{'tavg':{'mae':.01},'delta_t':{'mae':.02}}},'dev_common':{'standardized_mse':1.0}}
    value=copy.deepcopy(base)
    assert not runner.improvement_eligible(value,base)
    value['dev_common']['standardized_mse']=.5
    assert runner.improvement_eligible(value,base)
    value['old_validation']['metrics']['tavg']['mae']=.010501
    assert not runner.improvement_eligible(value,base)
    for role in ('test','test_common','old_test','sealed_test'):
        with pytest.raises(RuntimeError):runner.allowed_dataset({},role)
