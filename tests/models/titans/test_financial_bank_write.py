"""Escrituras por banco en lotes enmascarados y transporte de estados, sin ajuste."""

from dataclasses import asdict, replace

import pytest
import torch
from test_financial_adapter import api, raw_batch, setup
from test_frozen_financial import frozen_backend as frozen_backend
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.models.titans import MACConfig, MemoryConfig, TitansMAC
from mars_titan.models.titans.bank_readout import BankReadoutConfig
from mars_titan.models.titans.config import MemoryMultiscale
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.models.titans.transition_jacobian import (
    fast_state_dimension,
    fast_state_jacobian,
    fast_state_point,
    fast_state_transition,
)

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def test_old_three_field_recipe_still_uses_the_common_translation_identity():
    base, _ = setup()
    old = dict(routing="query", temperature=1.0, available=[True] * 3)
    cfg = replace(
        base.config, memory_multiscale=MemoryMultiscale(form="ema_bank"), memory_bank_readout=old
    )
    assert cfg.memory_bank_readout.write_gradient == "common_translation"
    assert cfg.identity()["memory_bank_readout"]["schema_version"] == 1
    assert "write_gradient" not in cfg.identity()["memory_bank_readout"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("rule", ["mean_partial", "per_bank"])
def test_financial_state_accounts_for_every_momentum_and_recovers_with_real_outer_gradients(
    device, dtype, graph, rule
):
    base, _ = setup()
    cfg = replace(
        base.config,
        memory_multiscale=MemoryMultiscale(form="ema_bank"),
        memory_bank_readout=BankReadoutConfig(write_gradient=rule),
        regional_fusion=RegionalFusionConfig(topology="ring") if graph else None,
    )
    model = api().FinancialPredictor(cfg, device=device, dtype=dtype).eval()
    parameter_rng = torch.get_rng_state().clone()
    paired = (
        api()
        .FinancialPredictor(
            replace(
                cfg,
                memory_bank_readout=replace(
                    cfg.memory_bank_readout,
                    write_gradient="mean_partial" if rule == "per_bank" else "per_bank",
                ),
            ),
            device=device,
            dtype=dtype,
        )
        .eval()
    )
    assert torch.equal(torch.get_rng_state(), parameter_rng)
    for name, value in model.named_parameters():
        assert torch.equal(value, dict(paired.named_parameters())[name])
    state = None
    work = []
    model.mac.memory.observer = lambda **values: work.append(len(values["momentum"]))
    try:
        for step in range(2):
            batch = api().DecisionBatch.from_corpus(
                raw_batch(at=1_609_459_200_000_000 + 10 * step),
                cfg.inputs,
                device=device,
                dtype=dtype,
            )
            state = (
                model.initial_state(batch.flow_ids, differentiable=True) if state is None else state
            )
            result = model.prepare(batch, state, differentiable=True)
            state = result.next_state
    finally:
        model.mac.memory.observer = None
    assert work == [6, 6]
    usage = model.state_usage(state)
    assert usage["tensor_bytes_per_flow"] == (98320 if dtype == torch.float64 else 49168)
    assert usage["tensor_bytes"] == 2 * usage["tensor_bytes_per_flow"]
    assert model.operation_counts(2)["bank_momentum_updates"] == 12
    assert model.operation_counts(2)["bank_gradient_matrices"] == 12
    result.point_predictions.square().sum().backward()
    gradient = model.mac.memory.bank_readout.addresses.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    model.requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    consumer = FrozenFinancialConsumer(model)
    restored = model.restore_state(model.export_state_cpu(state), device=device)
    next_batch = api().DecisionBatch.from_corpus(
        raw_batch(at=1_609_459_200_000_020), cfg.inputs, device=device, dtype=dtype
    )
    a = consumer.prepare(next_batch, state, context_id="a" * 64)
    b = consumer.prepare(next_batch, restored, context_id="a" * 64)
    assert torch.equal(a.point_predictions, b.point_predictions)
    assert a.next_state.mac.memory.steps.tolist() == [3, 3]
    assert len(a.next_state.mac.memory.momentum) == 6
    consumer.verify()


@pytest.mark.parametrize("device", DEVICES)
def test_dense_mac_diagnostic_includes_all_new_momentum_coordinates(device):
    cfg = MACConfig(
        memory=MemoryConfig(
            2,
            depth=1,
            bank_readout=BankReadoutConfig(write_gradient="per_bank"),
            multiscale=MemoryMultiscale(form="ema_bank"),
        ),
        heads=1,
        max_segment=1,
        persistent_tokens=1,
    )
    model = TitansMAC(cfg, device=device, dtype=torch.float64).eval()
    state = model.initial_state(1)
    token = torch.tensor([[[0.2, -0.3]]], dtype=torch.float64, device=device)
    point = fast_state_point(state).detach()
    assert fast_state_dimension(cfg) == point.numel() == 24
    transition = fast_state_transition(model, token, state)
    jacobian = fast_state_jacobian(model, token, state)
    assert jacobian.shape == (24, 24)
    direction = torch.linspace(-0.2, 0.4, 24, dtype=torch.float64, device=device)
    with sdpa_kernel(SDPBackend.MATH):
        numeric = (
            transition(point + 1e-6 * direction) - transition(point - 1e-6 * direction)
        ) / 2e-6
        _, following = model(token, state, differentiable=True)
        torch.testing.assert_close(transition(point), fast_state_point(following), atol=0, rtol=0)
    torch.testing.assert_close(jacobian @ direction, numeric, atol=1e-8, rtol=1e-6)


def test_full_serialized_bank_configuration_is_accepted_by_financial_config():
    base, _ = setup()
    reader = BankReadoutConfig(write_gradient="per_bank")
    cfg = replace(
        base.config,
        memory_multiscale=asdict(MemoryMultiscale(form="ema_bank")),
        memory_bank_readout=asdict(reader),
    )
    assert cfg.identity()["memory_bank_readout"] == reader.identity()


@pytest.mark.parametrize("device", DEVICES)
def test_declared_bank_write_arms_use_the_real_chronological_factory_without_optimizer(device):
    import json
    from pathlib import Path

    from mars_titan.training.financial_run import parameter_roles
    from mars_titan.training.titans_walk_forward import _predictor

    declaration = json.loads(Path("configs/experiments/bank-write-v1.json").read_text())
    recipe = json.loads(Path(declaration["base_recipe"]).read_text())
    base, _ = setup()
    for arm in declaration["arms"].values():
        model, pairing = _predictor(
            recipe, base.config.inputs, "mac_online", 42, device, changes=arm["predictor_changes"]
        )
        assert (
            pairing is None
            and "mac.memory.bank_readout.addresses" in parameter_roles(model)["shared"]
        )
        batch = api().DecisionBatch.from_corpus(
            raw_batch(), model.config.inputs, device=device, dtype=torch.float32
        )
        result = model.prepare(batch, model.initial_state(batch.flow_ids))
        assert result.quantiles.shape == (2, 5) and torch.isfinite(result.quantiles).all()
        assert len(result.next_state.mac.memory.momentum) == 6
        assert result.next_state.mac.memory.steps.tolist() == [1, 1]
