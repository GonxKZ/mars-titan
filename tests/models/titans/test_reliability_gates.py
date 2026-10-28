"""Puertas de Titans con la fiabilidad madura de #791: paridad, fórmula, cotas y validación."""

import json
import math
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
from test_financial_adapter import raw_batch, setup

from mars_titan.models.titans.financial import DecisionBatch, FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.models.titans.neural_memory import NeuralMemory
from mars_titan.models.titans.reliability_gates import (
    MATURE_RELIABILITY,
    MatureReliability,
    ReliabilityGates,
)
from mars_titan.training.search_cases import checked_predictor_changes

START, DAY = 1_609_459_200_000_000, 86_400_000_000
OPTIONS = [
    dict(),
    dict(source="shuffled_in_time"),
    dict(source="titans_surprise"),
    dict(independent=True),
    dict(consumers=("forget",), time_constants=(2,)),
]
IDS = ["mature", "shuffled", "surprise", "independent", "forget_only"]


def steps(model, count):
    spec = model.config.inputs
    return [
        DecisionBatch.from_corpus(raw_batch(at=START + i * DAY), spec, dtype=torch.float64)
        for i in range(count)
    ]


def gated(control, **options):
    config = replace(control.config, mature_reliability=MatureReliability(**options))
    return FinancialPredictor(config, dtype=torch.float64)


def signals(model, count, seed=5):
    config = model.config.mature_reliability
    shape = (2, len(config.consumers), len(config.time_constants))
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.rand(shape, generator=generator, dtype=torch.float64) * 2 - 1 for _ in range(count)
    ]


def walk(model, batches, values=None, *, differentiable=False):
    state = model.initial_state(batches[0].flow_ids, differentiable=differentiable)
    prepared = []
    for index, batch in enumerate(batches):
        options = {} if values is None else dict(reliability=values[index])
        prepared.append(model.prepare(batch, state, differentiable=differentiable, **options))
        state = prepared[-1].next_state
    return prepared, state


@pytest.mark.parametrize("options", OPTIONS, ids=IDS)
def test_a_declared_signal_starts_exactly_at_the_reference(options):
    control, _ = setup()
    model = gated(control, **options)
    batches = steps(control, 3)
    reference, last = walk(control, batches)
    observed, final = walk(model, batches, signals(model, 3))
    for left, right in zip(reference, observed, strict=True):
        assert torch.equal(left.point_predictions, right.point_predictions)
    for name in ("weights", "momentum"):
        pairs = zip(getattr(last.mac.memory, name), getattr(final.mac.memory, name), strict=True)
        for left, right in pairs:
            assert torch.equal(left, right)
    assert torch.count_nonzero(model.reliability_gates.mixing) == 0
    assert all(item.surprise is None for item in reference)


def test_gradients_match_the_reference_and_reach_the_mixing_of_each_gate():
    control, _ = setup()
    model = gated(control)
    batches = steps(control, 3)
    for engine, values in ((control, None), (model, signals(model, 3))):
        prepared, _ = walk(engine, batches, values, differentiable=True)
        torch.stack([item.point_predictions for item in prepared]).sum().backward()
    gradients = dict(model.named_parameters())
    mixing = gradients.pop("reliability_gates.mixing").grad
    assert gradients.keys() == dict(control.named_parameters()).keys()
    for name, parameter in control.named_parameters():
        other = gradients[name].grad
        assert (parameter.grad is None) == (other is None), name
        assert parameter.grad is None or torch.equal(parameter.grad, other), name
    assert mixing.shape == (2, 3) and bool((mixing != 0).all())


def test_a_signal_inside_the_margin_gives_no_gradient_to_the_mixing():
    control, _ = setup()
    model = gated(control, margin=0.5)
    batches = steps(control, 2)
    quiet = [torch.full((2, 2, 3), value, dtype=torch.float64) for value in (0.5, -0.3)]
    prepared, _ = walk(model, batches, quiet, differentiable=True)
    torch.stack([item.point_predictions for item in prepared]).sum().backward()
    assert torch.count_nonzero(model.reliability_gates.mixing.grad) == 0


def capture(monkeypatch):
    seen = []
    monkeypatch.setattr(NeuralMemory, "observer", staticmethod(lambda **v: seen.append(v)))
    return seen


def test_the_signal_shifts_the_write_and_forget_logits_by_the_declared_amount(monkeypatch):
    control, batch = setup()
    model = gated(control, margin=0.1, bound=0.5)
    with torch.no_grad():
        model.reliability_gates.mixing.copy_(
            torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], dtype=torch.float64)
        )
    model._seal_parameters()
    value = torch.tensor([[0.3, -0.05, 0.9], [-0.6, 0.2, 0.08]], dtype=torch.float64)
    seen = capture(monkeypatch)
    with torch.no_grad():
        control.prepare(batch, control.initial_state(batch.flow_ids))
        model.prepare(batch, model.initial_state(batch.flow_ids), reliability=value.expand(2, 2, 3))
    reference, shifted = seen
    # Con margen 0,1 quedan u = [0,2, 0, 0,8] y [−0,5, 0,1, 0], así que la escritura se desplaza
    # 0,5 · tanh(0,6) y el olvido 0,5 · tanh(0,3) en unidades de logit.
    theta_max = control.mac.memory.config.theta_max
    write = torch.logit(shifted["theta"] / theta_max) - torch.logit(reference["theta"] / theta_max)
    forget = torch.logit(shifted["alpha"]) - torch.logit(reference["alpha"])
    torch.testing.assert_close(write, torch.full_like(write, 0.5 * math.tanh(0.6)))
    torch.testing.assert_close(forget, torch.full_like(forget, 0.5 * math.tanh(0.3)))
    assert torch.equal(shifted["eta"], reference["eta"])


def test_each_shift_stays_within_the_bound_and_vanishes_inside_the_margin():
    config = MatureReliability(margin=0.2, bound=0.75)
    gates = ReliabilityGates(config, dtype=torch.float64)
    with torch.no_grad():
        gates.mixing.fill_(1e6)
    extreme = gates(torch.tensor([[[1.0] * 3, [-1.0] * 3]], dtype=torch.float64))
    assert extreme["write"].item() == 0.75 and extreme["forget"].item() == -0.75
    quiet = gates(torch.tensor([[[0.2, -0.2, 0.1], [0.0, 0.15, -0.05]]], dtype=torch.float64))
    assert quiet["write"].item() == 0.0 and quiet["forget"].item() == 0.0
    assert extreme["write"].shape == (1, 1)
    only = ReliabilityGates(MatureReliability(consumers=("write",)), dtype=torch.float64)
    assert set(only(torch.zeros(3, 1, 3, dtype=torch.float64))) == {"write"}


def test_the_surprise_control_returns_the_associative_loss_of_each_row(monkeypatch):
    control, batch = setup()
    model = gated(control, source="titans_surprise")
    seen = capture(monkeypatch)
    state = model.initial_state(batch.flow_ids)
    prepared = model.prepare(batch, state, reliability=torch.zeros(2, 2, 3, dtype=torch.float64))
    surprise = prepared.surprise
    assert surprise.shape == (2,) and not surprise.requires_grad
    assert bool((surprise >= 0).all()) and bool(torch.isfinite(surprise).all())
    torch.testing.assert_close(surprise.sum(), seen[0]["loss"].detach(), rtol=1e-12, atol=0)
    other = gated(control)
    zero = torch.zeros(2, 2, 3, dtype=torch.float64)
    plain = other.prepare(batch, other.initial_state(batch.flow_ids), reliability=zero)
    assert plain.surprise is None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "necesita su señal"),
        (torch.zeros(3, 2, 3, dtype=torch.float64), "no corresponde"),
        (torch.zeros(2, 1, 3, dtype=torch.float64), "no corresponde"),
        (torch.zeros(2, 2, 2, dtype=torch.float64), "no corresponde"),
        (torch.zeros(2, 2, 3, dtype=torch.float32), "no corresponde"),
        (torch.zeros(2, 2, 3, dtype=torch.float64, requires_grad=True), "no corresponde"),
        ([[0.0] * 3] * 2, "no corresponde"),
        (torch.full((2, 2, 3), 1.5, dtype=torch.float64), "finita y estar"),
        (torch.full((2, 2, 3), math.nan, dtype=torch.float64), "finita y estar"),
    ],
)
def test_prepare_rejects_a_missing_or_malformed_signal(value, message):
    control, batch = setup()
    model = gated(control)
    with pytest.raises(ValueError, match=message):
        model.prepare(batch, model.initial_state(batch.flow_ids), reliability=value)


def test_a_predictor_without_the_component_refuses_a_signal():
    control, batch = setup()
    with pytest.raises(ValueError, match="no declara"):
        control.prepare(
            batch,
            control.initial_state(batch.flow_ids),
            reliability=torch.zeros(2, 2, 3, dtype=torch.float64),
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (dict(source="titans_loss"), "fuente"),
        (dict(time_constants=()), "time_constants"),
        (dict(time_constants=(5, 1)), "time_constants"),
        (dict(time_constants=(1, 1)), "time_constants"),
        (dict(time_constants=(0,)), "time_constants"),
        (dict(time_constants=(253,)), "time_constants"),
        (dict(time_constants=(1.0,)), "time_constants"),
        (dict(time_constants=(1, 2, 3, 4, 5)), "time_constants"),
        (dict(time_constants="1"), "time_constants"),
        (dict(consumers=()), "consumers"),
        (dict(consumers=("forget", "write")), "consumers"),
        (dict(consumers=("write", "write")), "consumers"),
        (dict(consumers=("read",)), "consumers"),
        (dict(consumers="write"), "consumers"),
        (dict(independent=True, consumers=("write",)), "independientes"),
        (dict(independent=1), "independientes"),
        (dict(seed=-1), "semilla"),
        (dict(seed=2**32), "semilla"),
        (dict(seed=True), "semilla"),
        (dict(margin=-0.01), "margen"),
        (dict(margin=0.51), "margen"),
        (dict(margin=math.nan), "margen"),
        (dict(margin=True), "margen"),
        (dict(bound=0.0), "cota"),
        (dict(bound=4.01), "cota"),
        (dict(bound=-1.0), "cota"),
        (dict(bound=math.inf), "cota"),
    ],
)
def test_the_configuration_rejects_undeclared_values(options, message):
    with pytest.raises(ValueError, match=message):
        MatureReliability(**options)


def test_the_configuration_records_its_role_and_normalizes_its_lists():
    config = MatureReliability(time_constants=[1, 5], consumers=["write", "forget"], margin=0)
    assert config.time_constants == (1, 5) and config.consumers == ("write", "forget")
    assert config.margin == 0.0 and type(config.margin) is float
    assert config.channels == 1
    identity = config.identity()
    assert identity["name"] == MATURE_RELIABILITY
    assert identity["role"] == "shared_signal_variant"
    assert identity["time_constants"] == [1, 5] and identity["consumers"] == ["write", "forget"]
    # Cada control conserva su propio papel, no solo el conjunto de papeles.
    roles = {
        source: MatureReliability(source=source).identity()["role"]
        for source in ("shuffled_in_time", "titans_surprise")
    }
    assert roles == dict(
        shuffled_in_time="shuffled_in_time_control", titans_surprise="titans_surprise_control"
    )
    independent = MatureReliability(independent=True).identity()["role"]
    assert independent == "independent_signals_per_consumer_control"
    assert MatureReliability(independent=True).channels == 2
    scores = {MatureReliability(source=s).identity()["score"] for s in ("shuffled_in_time",)}
    assert scores == {"score_of_a_uniformly_drawn_earlier_or_current_maturity_event"}
    with pytest.raises(ValueError, match="configuración"):
        ReliabilityGates(dict(), dtype=torch.float64)


def test_the_predictor_declares_the_component_only_with_online_memory():
    control, _ = setup()
    spec = control.config.inputs
    values = {
        field.name: getattr(MatureReliability(), field.name) for field in fields(MatureReliability)
    }
    parsed = FinancialConfig(spec, variant="mac_online", mature_reliability=values)
    assert parsed.mature_reliability == MatureReliability()
    assert parsed.identity()["mature_reliability"] == MatureReliability().identity()
    assert "mature_reliability" not in FinancialConfig(spec, variant="mac_online").identity()
    with pytest.raises(ValueError, match="todos sus campos"):
        FinancialConfig(
            spec, variant="mac_online", mature_reliability=dict(source="titans_surprise")
        )
    for variant in ("transformer_direct", "mac_disabled", "mac_frozen"):
        with pytest.raises(ValueError, match="mac_online"):
            FinancialConfig(spec, variant=variant, mature_reliability=MatureReliability())
    with pytest.raises(ValueError, match="mac_online"):
        FinancialConfig(spec, variant="mac_online", mature_reliability="shared")
    assert checked_predictor_changes(dict(mature_reliability=values)) == dict(
        mature_reliability=values
    )


def test_the_signal_is_refused_where_it_cannot_reach_the_gates():
    control, batch = setup()
    model = gated(control)
    with pytest.raises(ValueError, match="control local"):
        FinancialPredictor(
            model.config,
            local_control=MACProjectionConfig(mode="disabled", rank=1, grid_size=4),
            dtype=torch.float64,
        )
    with pytest.raises(ValueError, match="fiabilidad madura"):
        FrozenFinancialConsumer(model.eval().requires_grad_(False))
    frozen, _ = setup("mac_frozen")
    state = frozen.initial_state(batch.flow_ids)
    token = torch.zeros(2, 1, frozen.config.hidden_size, dtype=torch.float64)
    for options in (dict(gate_shift={}), dict(surprise=[])):
        with pytest.raises(ValueError, match="memoria online"):
            frozen.mac(token, state.mac, **options)


def test_the_declared_arms_build_and_each_control_changes_one_thing():
    root = Path(__file__).resolve().parents[3]
    path = root / "configs/experiments/mature-reliability-signal-v1.json"
    declaration = json.loads(path.read_text(encoding="utf-8"))
    assert declaration["default_enabled"] is False and declaration["final_test_access"] is False
    control, _ = setup()
    reference = dict(control.named_parameters())
    roles, identities = {}, set()
    for name, arm in declaration["arms"].items():
        changes = arm["predictor_changes"]
        assert checked_predictor_changes(changes or None) == (changes or None)
        model = FinancialPredictor(replace(control.config, **changes), dtype=torch.float64)
        identities.add(json.dumps(model.config.identity(), sort_keys=True))
        parameters = dict(model.named_parameters())
        for key, value in reference.items():
            assert torch.equal(parameters[key], value), (name, key)
        if changes:
            roles[name] = model.config.mature_reliability.role
    assert len(identities) == len(declaration["arms"])
    assert roles == {
        "shared_signal": "shared_signal_variant",
        "shuffled_in_time": "shuffled_in_time_control",
        "titans_surprise": "titans_surprise_control",
        "independent_signals": "independent_signals_per_consumer_control",
    }
    arms = declaration["arms"]
    for contrast in declaration["contrasts"]:
        assert contrast["variant"] == "shared_signal" and contrast["control"] in arms
    shared = arms["shared_signal"]["predictor_changes"]["mature_reliability"]
    changed = {}
    for name in ("shuffled_in_time", "titans_surprise", "independent_signals"):
        other = arms[name]["predictor_changes"]["mature_reliability"]
        changed[name] = {key for key in shared if shared[key] != other[key]}
    assert changed == {
        "shuffled_in_time": {"source"},
        "titans_surprise": {"source"},
        "independent_signals": {"independent"},
    }
