"""CPU contract tests; this pilot deliberately reuses the external frozen prior."""
import copy
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

from hnuter_dit.sdf_model import SDFPoseDiT, sample as sample_base
from hnuter_dit.sdf_mapping import CHANNELS, GRID
from hnuter_dit.task_adapter import FrozenSDFTaskAdapter, sample_task, DEFAULT_PHASE


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_model(output_residual=True):
    torch.manual_seed(97)
    base = SDFPoseDiT(dimension=32, depth=4, heads=4)
    # A trained base has nonzero output weights; zero initialized output would
    # trivially hide adapter errors and block every adapter gradient.
    with torch.no_grad():
        base.output.weight.normal_(std=.07)
        base.output.bias.normal_(std=.03)
        base.mean.normal_(std=.02)
        base.std.uniform_(.1,.2)
        for block in base.blocks:
            block.modulation[-1].weight.normal_(std=.01)
    return FrozenSDFTaskAdapter(base,output_residual=output_residual)


def inputs(model, batch=2):
    generator = torch.Generator().manual_seed(321)
    return (torch.randn(batch,6,6,generator=generator), torch.tensor([37]*batch),
            torch.randn(batch,6,generator=generator),
            torch.randn(batch,9,model.dimension,generator=generator),
            torch.randn(batch,24,generator=generator))


def snapshot(model):
    return {name:value.clone() for name,value in model.state_dict().items()}


def test_enabled_initialization_and_disabled_delegate_exactly():
    model = make_model(); x,t,target,memory,task = inputs(model)
    base_state = snapshot(model.base)
    with torch.no_grad():
        expected = model.base.forward_cached(x,t,target,memory)
        assert torch.equal(expected,model.forward_cached(x,t,target,memory,task))
    # Autograd changes MHA kernel dispatch after a zero residual; allow only
    # float rounding here while the inference identity remains bitwise.
    torch.testing.assert_close(expected,model.forward_cached(x,t,target,memory,task),rtol=1e-6,atol=2e-6)
    with patch.object(model.base,'forward_cached',wraps=model.base.forward_cached) as forward:
        disabled = model.forward_cached(x,t,target,memory,None,phase='ignored',enabled=False)
        forward.assert_called_once_with(x,t,target,memory)
    assert torch.equal(expected,disabled)
    assert all(torch.equal(value,model.base.state_dict()[name]) for name,value in base_state.items())
    assert all(not p.requires_grad and p.grad is None for p in model.base.parameters())
    assert model.mean.data_ptr() == model.base.mean.data_ptr()
    assert model.std.data_ptr() == model.base.std.data_ptr()
    assert model.adapter_configuration()['phase_semantics'].startswith('soft control-support')


def test_only_adapter_updates_and_gradients_reach_task_branch():
    model = make_model(); model.train()
    assert model.training and not model.base.training
    x,t,target,memory,task = inputs(model)
    before = snapshot(model.base)
    expected = model.base.forward_cached(x,t,target,memory).detach()
    parameters = list(model.adapter_parameters())
    assert parameters and not set(map(id,parameters)) & set(map(id,model.base.parameters()))
    optimizer = torch.optim.AdamW(parameters,lr=.003,weight_decay=0.)
    loss = (model.forward_cached(x,t,target,memory,task)-.7).square().mean()
    loss.backward()
    # At step zero the zero projections receive gradients; deeper branch
    # layers start receiving useful gradients once these gates are updated.
    for branch in model.branches.values():
        assert branch.zero.weight.grad is not None
        assert torch.isfinite(branch.zero.weight.grad).all()
        assert branch.zero.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    changed = model.forward_cached(x,t,target,memory,task)
    assert not torch.equal(changed,expected)
    (changed-.7).square().mean().backward()
    assert model.task_encoder[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())
    optimizer.step()
    assert all(torch.equal(value,model.base.state_dict()[name]) for name,value in before.items())
    assert torch.equal(expected,model.forward_cached(x,t,target,memory,None,enabled=False))


def test_zero_phase_suppresses_trained_residual_and_batch_phase_works():
    model = make_model(); x,t,target,memory,task = inputs(model)
    with torch.no_grad():
        for branch in model.branches.values():
            branch.zero.weight.normal_(std=.04)
            branch.zero.bias.fill_(.02)
    with torch.no_grad():
        expected = model.base.forward_cached(x,t,target,memory)
        assert torch.equal(expected,model.forward_cached(x,t,target,memory,task,phase=torch.zeros(6)))
    phase = torch.tensor(DEFAULT_PHASE).expand(2,-1)
    assert torch.equal(model.forward_cached(x,t,target,memory,task),
                       model.forward_cached(x,t,target,memory,task,phase=phase))
    alternate = task+torch.linspace(-1,1,24)[None]
    assert not torch.equal(model.forward_cached(x,t,target,memory,task),
                           model.forward_cached(x,t,target,memory,alternate))


@pytest.mark.parametrize('bad_task', [None, torch.zeros(24), torch.zeros(2,23),
    torch.zeros(2,24,dtype=torch.long), torch.full((2,24),float('nan'))])
def test_invalid_task_rejected(bad_task):
    model = make_model(); x,t,target,memory,_ = inputs(model)
    with pytest.raises(ValueError,match='task must'):
        model.forward_cached(x,t,target,memory,bad_task)


@pytest.mark.parametrize('bad_phase', [torch.zeros(5),torch.zeros(2,6,1),
    torch.zeros(3,6),torch.full((6,),float('nan')),[-.1,0,0,0,0,0],[0,0,0,0,0,1.1]])
def test_invalid_mask_rejected(bad_phase):
    model = make_model(); x,t,target,memory,task = inputs(model)
    with pytest.raises(ValueError,match='phase'):
        model.forward_cached(x,t,target,memory,task,phase=bad_phase)


def test_independent_adapter_checkpoint_never_contains_or_changes_base():
    first = make_model()
    # A second task may share the same frozen base but has separate branch weights.
    second = FrozenSDFTaskAdapter(first.base)
    assert set(map(id,first.adapter_parameters())).isdisjoint(map(id,second.adapter_parameters()))
    initial_second = second.adapter_state_dict()
    with torch.no_grad():
        first.branches['3'].zero.weight.fill_(.05)
    assert all(torch.equal(value,second.adapter_state_dict()[key]) for key,value in initial_second.items())
    saved = first.adapter_state_dict();base_before=snapshot(first.base)
    assert not any(key.startswith('base.') for key in saved)
    second.load_adapter_state_dict(saved)
    assert all(torch.equal(value,second.adapter_state_dict()[key]) for key,value in saved.items())
    assert all(torch.equal(value,first.base.state_dict()[key]) for key,value in base_before.items())
    wrong = copy.deepcopy(saved);wrong['base.mean']=first.base.mean.clone()
    with pytest.raises(ValueError,match='keys'):
        second.load_adapter_state_dict(wrong)
    wrong = copy.deepcopy(saved);wrong['default_phase'][0]=float('nan')
    with pytest.raises(ValueError,match='invalid adapter tensor'):
        second.load_adapter_state_dict(wrong)
    assert all(torch.equal(value,second.adapter_state_dict()[key]) for key,value in saved.items())


def test_frozen_encode_and_raw_cached_sampler_match_base():
    model = make_model()
    fields = torch.randn(1,4,len(CHANNELS),GRID,GRID,GRID,requires_grad=True)
    ages = torch.tensor([[-.6,-.4,-.2,0.]])
    target = torch.tensor([[2.,.1,.05,.1,-.1,.2]])
    task = torch.randn(1,24)
    memory = model.encode(fields,ages)
    assert not memory.requires_grad
    assert torch.equal(memory,model.base.encode(fields.detach(),ages))
    expected = sample_base(model.base,fields.detach(),ages,target,seed=114,steps=3)
    disabled = sample_task(model,fields,ages,target,None,seed=114,steps=3,enabled=False)
    initialized = sample_task(model,None,None,target,task,seed=114,steps=3,memory=memory)
    assert expected.shape == (1,6,6)
    assert torch.equal(expected,disabled)
    assert torch.equal(expected,initialized)
    for bad_steps in [0,101,True,1.5]:
        with pytest.raises(ValueError,match='steps'):
            sample_task(model,None,None,target,task,memory=memory,steps=bad_steps)
    with pytest.raises(ValueError,match='fields and ages'):
        sample_task(model,None,None,target,task)


def test_optional_output_residual_extends_frozen_readout_range_without_base_changes():
    model=make_model();x,t,target,memory,task=inputs(model)
    assert model.adapter_configuration()['output_residual'] is True
    assert 'new SDF-pilot extension' in model.adapter_configuration()['output_residual_design']
    before=snapshot(model.base)
    with torch.no_grad():
        expected=model.base.forward_cached(x,t,target,memory)
        model.output_head[-1].bias.fill_(20.)
        expanded=model.forward_cached(x,t,target,memory,task,phase=torch.ones(6))
        torch.testing.assert_close(expanded,expected+20.,rtol=0,atol=2e-6)
        assert torch.equal(expected,model.forward_cached(x,t,target,memory,task,enabled=False))
        assert torch.equal(expected,model.forward_cached(x,t,target,memory,task,phase=torch.zeros(6)))
    assert all(torch.equal(value,model.base.state_dict()[name]) for name,value in before.items())
    block_only=FrozenSDFTaskAdapter(model.base,output_residual=False)
    assert block_only.output_head is None
    assert block_only.adapter_configuration()['output_residual'] is False
    with torch.no_grad():
        assert torch.equal(expected,block_only.forward_cached(x,t,target,memory,task))
    assert not any(k.startswith('output_head.') for k in block_only.adapter_state_dict())
    with pytest.raises(ValueError,match='keys'):
        block_only.load_adapter_state_dict(model.adapter_state_dict())
    with pytest.raises(ValueError,match='output_residual'):
        FrozenSDFTaskAdapter(model.base,output_residual='yes')


def test_output_head_is_trainable_and_checkpointed_separately():
    model=make_model();x,t,target,memory,task=inputs(model)
    model.forward_cached(x,t,target,memory,task).square().mean().backward()
    assert model.output_head[-1].weight.grad is not None
    assert torch.isfinite(model.output_head[-1].weight.grad).all()
    assert model.output_head[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.base.parameters())
    assert any(k.startswith('output_head.') for k in model.adapter_state_dict())
