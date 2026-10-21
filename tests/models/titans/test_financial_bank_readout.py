"""Lectura histórica dentro del predictor y de la recuperación congelada, sin ajuste."""

from dataclasses import asdict, replace

import pytest
import torch
from test_bank_readout import config
from test_financial_adapter import api, raw_batch, setup
from test_frozen_financial import frozen_backend as frozen_backend

from mars_titan.models.titans.config import MemoryMultiscale
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.training.search_cases import checked_predictor_changes

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def test_bank_readout_travels_through_existing_recipe_and_has_its_own_identity():
    base, _ = setup()
    options = dict(
        memory_multiscale=asdict(MemoryMultiscale(form="ema_bank")),
        memory_bank_readout=asdict(config()),
    )
    assert checked_predictor_changes(options) == options
    changed = replace(base.config, **options)
    assert "memory_bank_readout" not in base.config.identity()
    assert changed.identity()["memory_bank_readout"] == config().identity()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("graph", [False, True])
def test_financial_readout_keeps_tensor_state_size_recovers_and_has_real_gradients(
    device, dtype, graph
):
    base, _ = setup()
    cfg = replace(base.config, memory_multiscale=MemoryMultiscale(form="ema_bank"))
    if graph:
        cfg = replace(cfg, regional_fusion=RegionalFusionConfig(topology="ring"))
    control = api().FinancialPredictor(cfg, device=device, dtype=dtype).eval()
    original_rng = torch.random.get_rng_state().clone()
    cfg = replace(cfg, memory_bank_readout=config())
    model = api().FinancialPredictor(cfg, device=device, dtype=dtype).eval()
    assert torch.equal(original_rng, torch.random.get_rng_state())
    assert model._tensor_bytes_per_flow() == control._tensor_bytes_per_flow()
    for name, value in control.named_parameters():
        assert torch.equal(value, dict(model.named_parameters())[name]), name
    state = None
    for step in range(3):
        batch = api().DecisionBatch.from_corpus(
            raw_batch(at=1_609_459_200_000_000 + 10 * step), cfg.inputs, device=device, dtype=dtype
        )
        state = model.initial_state(batch.flow_ids, differentiable=True) if state is None else state
        result = model.prepare(batch, state, differentiable=True)
        state = result.next_state
    result.point_predictions.square().sum().backward()
    gradient = model.mac.memory.bank_readout.addresses.grad
    assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
    assert state.mac.memory.steps.tolist() == [3, 3]
    model.eval().requires_grad_(False)
    for value in model.parameters():
        value.grad = None
    engine = FrozenFinancialConsumer(model)
    restored = model.restore_state(model.export_state_cpu(state), device=device)
    following = api().DecisionBatch.from_corpus(
        raw_batch(at=1_609_459_200_000_030), cfg.inputs, device=device, dtype=dtype
    )
    a = engine.prepare(following, state, context_id="a" * 64)
    b = engine.prepare(following, restored, context_id="a" * 64)
    torch.testing.assert_close(a.point_predictions, b.point_predictions, rtol=0, atol=0)
    engine.verify()


def test_frozen_reader_rejects_changed_routing_and_invalid_parent_checkpoint():
    import copy

    base, _ = setup()
    cfg = replace(
        base.config,
        memory_multiscale=MemoryMultiscale(form="ema_bank"),
        memory_bank_readout=config(),
    )
    model = api().FinancialPredictor(cfg, dtype=torch.float64).eval().requires_grad_(False)
    saved = copy.deepcopy(model.state_dict())
    invalid = copy.deepcopy(saved)
    invalid["head.weight"] += 1
    invalid["mac.memory.bank_readout.available"][0] = False
    with pytest.raises(ValueError):
        model.load_state_dict(invalid)
    assert torch.equal(model.state_dict()["head.weight"], saved["head.weight"])
    engine = FrozenFinancialConsumer(model)
    model.mac.memory.bank_readout.config = replace(
        model.mac.memory.bank_readout.config, temperature=2
    )
    with pytest.raises(ValueError, match="identidad"):
        engine.verify()


@pytest.mark.parametrize("device", DEVICES)
def test_bank_work_counters_match_the_memory_calls_actually_executed(device):
    base, _ = setup()
    cfg = replace(
        base.config,
        memory_multiscale=MemoryMultiscale(form="ema_bank"),
        memory_bank_readout=config(),
    )
    model = api().FinancialPredictor(cfg, device=device, dtype=torch.float64).eval()
    batch = api().DecisionBatch.from_corpus(
        raw_batch(), cfg.inputs, device=device, dtype=torch.float64
    )
    observations = []
    hook = model.mac.memory.bank_readout.register_forward_hook(
        lambda _, args, result: observations.append(args[0].shape[0] * args[0].shape[1] * 3)
    )
    try:
        result = model.prepare(batch, model.initial_state(batch.flow_ids))
    finally:
        hook.remove()
    counts = model.operation_counts(2)
    assert counts["bank_function_evaluations"] == sum(observations) == 18
    assert counts["bank_query_routes"] == 6
    assert counts["bank_matrix_updates"] == 12
    assert result.next_state.mac.memory.steps.tolist() == [1, 1]


@pytest.mark.parametrize("device", DEVICES)
def test_declared_bank_arms_use_the_existing_factory_without_optimizer(device):
    import json
    from pathlib import Path

    from mars_titan.training.financial_run import parameter_roles
    from mars_titan.training.titans_walk_forward import _predictor

    declaration = json.loads(Path("configs/experiments/bank-readout-v1.json").read_text())
    recipe = json.loads(Path(declaration["base_recipe"]).read_text())
    base, _ = setup()
    for name, arm in declaration["arms"].items():
        model, pairing = _predictor(
            recipe, base.config.inputs, "mac_online", 42, device, changes=arm["predictor_changes"]
        )
        assert pairing is None
        batch = api().DecisionBatch.from_corpus(
            raw_batch(), model.config.inputs, device=device, dtype=torch.float32
        )
        result = model.prepare(batch, model.initial_state(batch.flow_ids))
        assert torch.isfinite(result.quantiles).all() and result.quantiles.shape == (2, 5)
        assert result.next_state.mac.memory.steps.tolist() == [1, 1]
        if name != "ema_weight_mean":
            assert "mac.memory.bank_readout.addresses" in parameter_roles(model)["shared"]
