"""La disposición explícita evita cambiar QKV al congelar los pesos, sin alterar el legado."""

import copy
from dataclasses import replace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.models.titans import MACConfig, MemoryConfig, TitansMAC
from mars_titan.training.titans_walk_forward import unfused_attention

LAYOUT = "sequence_contiguous_v1"


def config(**options):
    return MACConfig(
        memory=MemoryConfig(dim=64, depth=2, max_batch=64, max_tokens=3),
        heads=4,
        max_segment=3,
        **options,
    )


def test_new_layout_has_a_distinct_identity_and_default_checkpoints_remain_readable():
    legacy = TitansMAC(config())
    old = copy.deepcopy(legacy.state_dict())
    assert "attention_layout" not in old["_extra_state"]["configuration"]
    legacy.load_state_dict(old)
    explicit = config(attention_layout=LAYOUT)
    assert explicit.fingerprint() != config().fingerprint()
    assert explicit.identity()["attention_layout"] == LAYOUT
    with pytest.raises(ValueError):
        TitansMAC(explicit).load_state_dict(old)


@pytest.mark.parametrize("value", [None, False, {}, [], "unknown"])
def test_unknown_attention_layouts_fail_before_constructing_layers(value):
    with pytest.raises(ValueError, match="attention_layout"):
        config(attention_layout=value)


@pytest.mark.parametrize("mode", ["online", "frozen", "disabled"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_freezing_does_not_change_the_explicit_layout_trajectory(mode, dtype, device):
    core = TitansMAC(config(attention_layout=LAYOUT, memory_mode=mode), dtype=dtype, device=device)
    frozen = copy.deepcopy(core).requires_grad_(False)
    legacy = TitansMAC(config(memory_mode=mode), dtype=dtype, device=device)
    states = [m.initial_state(64) for m in (core, frozen, legacy)]
    generator = torch.Generator().manual_seed(662)
    tokens = torch.randn(5, 64, 3, 64, generator=generator, dtype=dtype).to(device)
    with unfused_attention(), sdpa_kernel(SDPBackend.MATH), torch.no_grad():
        for token in tokens:
            output = []
            for i, model in enumerate((core, frozen, legacy)):
                value, states[i] = model(token, states[i])
                output.append(value)
            for other in output[1:]:
                torch.testing.assert_close(output[0], other, rtol=0, atol=0)
            for field in ("weights", "momentum", "levels"):
                expected = getattr(states[0].memory, field)
                for state in states[1:]:
                    for left, right in zip(expected, getattr(state.memory, field), strict=True):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
    saved = core.export_state(states[0])
    restored = core.restore_state(saved)
    with unfused_attention(), sdpa_kernel(SDPBackend.MATH), torch.no_grad():
        actual, after = core(tokens[0], restored)
        expected, _ = core(tokens[0], states[0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(after.memory.steps, states[0].memory.steps + (3 if mode == "online" else 0))
    assert all(p.grad is None for model in (core, frozen, legacy) for p in model.parameters())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_layout_keeps_gradients_through_attention_and_associative_updates(device):
    small = MACConfig(memory=MemoryConfig(dim=4, depth=2, max_tokens=2), heads=2, max_segment=2)
    old = TitansMAC(small, dtype=torch.float64, device=device)
    new = TitansMAC(replace(small, attention_layout=LAYOUT), dtype=torch.float64, device=device)
    values = torch.arange(16, dtype=torch.float64, device=device).reshape(2, 2, 4) / 20
    gradients, predictions = [], []
    with unfused_attention(), sdpa_kernel(SDPBackend.MATH):
        for core in (old, new):
            x = values.clone().requires_grad_()
            result, _ = core(x, core.initial_state(2, differentiable=True), differentiable=True)
            predictions.append(result)
            gradients.append(torch.autograd.grad(result.square().sum(), (x, *core.parameters())))
    torch.testing.assert_close(*predictions, rtol=0, atol=0)
    for left, right in zip(*gradients, strict=True):
        assert torch.isfinite(right).all()
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)


def test_layout_does_not_consume_rng_or_change_initial_parameters():
    legacy = TitansMAC(config())
    rng = torch.get_rng_state().clone()
    changed = TitansMAC(config(attention_layout=LAYOUT))
    assert torch.equal(rng, torch.get_rng_state())
    for a, b in zip(legacy.parameters(), changed.parameters(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_legacy_default_keeps_its_attention_route_even_when_frozen(flag, device):
    core = TitansMAC(config(memory_mode="disabled"), device=device).eval().requires_grad_(flag)
    generator = torch.Generator().manual_seed(662)
    x = torch.randn(64, 3, 64, generator=generator).to(device)
    mask = torch.ones(3, 3, dtype=torch.bool, device=device).triu(1)
    with unfused_attention(), sdpa_kernel(SDPBackend.MATH), torch.no_grad():
        expected = core.attention(x, x, x, attn_mask=mask, need_weights=False)[0][:, -1]
        actual, _ = core(x, core.initial_state(64))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_financial_predictor_carries_the_layout_into_its_mac_and_identity():
    from test_financial_adapter import specification

    from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor

    legacy = FinancialConfig(specification(), variant="mac_online")
    assert "attention_layout" not in legacy.identity()
    declared = replace(legacy, attention_layout=LAYOUT)
    model = FinancialPredictor(declared)
    assert model.mac.config.attention_layout == LAYOUT
    assert model.get_extra_state()["configuration"]["attention_layout"] == LAYOUT
    with pytest.raises(ValueError, match="attention_layout"):
        replace(legacy, attention_layout="unknown")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def test_explicit_layout_preserves_c_penalty_gradients_and_the_committed_past(device):
    from test_financial_adapter import api, raw_batch
    from test_financial_control import configured, prepare

    legacy, _ = configured("penalty")
    legacy = legacy.to(device)
    new = api().FinancialPredictor(
        replace(legacy.config, attention_layout=LAYOUT),
        local_control=legacy.local_control.config,
        device=device,
        dtype=torch.float64,
    )
    batch = api().DecisionBatch.from_corpus(
        raw_batch(), new.config.inputs, device=device, dtype=torch.float64
    )
    prepared, gradients = [], []
    with unfused_attention():
        for model in (legacy, new):
            value = prepare(model, batch, model.initial_state(batch.flow_ids), differentiable=True)
            prepared.append(value)
            gradients.append(
                torch.autograd.grad(
                    value.local_control.penalty, model.mac.memory.eta_projection.weight
                )[0]
            )
    torch.testing.assert_close(
        prepared[0].point_predictions, prepared[1].point_predictions, rtol=0, atol=0
    )
    torch.testing.assert_close(*gradients, rtol=1e-10, atol=1e-12)
    assert gradients[1].abs().sum() > 0 and torch.isfinite(gradients[1]).all()

    new.eval().requires_grad_(False)
    state = new.initial_state(batch.flow_ids)
    with unfused_attention(), torch.no_grad():
        first = prepare(new, batch, state)
        saved = first.point_predictions.clone()
        for scale in (1.0, -100.0):
            future = raw_batch(at=batch.prediction_at[0] + 86_400_000_000, absent=False)
            future["inputs"]["news"] *= scale
            later = api().DecisionBatch.from_corpus(
                future, new.config.inputs, device=device, dtype=torch.float64
            )
            prepare(new, later, first.next_state)
            repeated = prepare(new, batch, state)
            torch.testing.assert_close(repeated.point_predictions, saved, rtol=0, atol=0)
            torch.testing.assert_close(first.point_predictions, saved, rtol=0, atol=0)
    assert state.mac.memory.steps.tolist() == [0, 0]
