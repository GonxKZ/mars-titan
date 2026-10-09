"""Titans-MAC financiero con `quantile_head_v1`, lectura episódica y consumidor congelado.

Las pruebas usan lotes técnicos en CPU. Calculan forward y gradientes, nunca pasos
de optimizador.
"""

import json
import math

import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, reader, snapshot, source
from test_financial_adapter import VARIANTS, raw_batch, specification

from mars_titan.models.quantile_head import (
    CONTRACT,
    MEDIAN_INDEX,
    QUANTILE_HEAD,
    QuantileHead,
    median,
    pinball_loss,
)
from mars_titan.models.titans.episodic_readout import apply_episodic_readout
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.financial import (
    DecisionBatch,
    FinancialConfig,
    FinancialPredictor,
    copy_paired_parameters,
)
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer


def build(variant="mac_online", *, head="scalar", historical=True, seed=42):
    spec = specification(historical=historical)
    config = FinancialConfig(spec, variant=variant, hidden_size=32, layers=1, seed=seed, head=head)
    model = FinancialPredictor(config, dtype=torch.float64)
    raw = raw_batch(absent=historical)
    if not historical:
        raw.pop("presence")
    return model, DecisionBatch.from_corpus(raw, spec, dtype=torch.float64)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("historical", [True, False])
def test_quantile_variant_keeps_the_scalar_trunk_and_identity_except_the_output(
    variant, historical
):
    scalar, batch = build(variant, historical=historical)
    quantile, _ = build(variant, head=QUANTILE_HEAD, historical=historical)
    left, right = dict(scalar.named_parameters()), dict(quantile.named_parameters())
    assert left.keys() == right.keys()
    for name, value in left.items():
        if not name.startswith("head."):
            assert torch.equal(value, right[name]), name
    assert isinstance(quantile.head, QuantileHead) and quantile.head.weight.shape == (5, 32)
    scalar_identity, quantile_identity = scalar.config.identity(), quantile.config.identity()
    assert scalar_identity["output"] == "scalar_corpus_target"
    assert "output_head" not in scalar_identity
    assert quantile_identity == {
        **scalar_identity,
        "output": QUANTILE_HEAD,
        "output_head": CONTRACT,
    }
    assert scalar._config_id() != quantile._config_id()
    first = scalar.prepare(batch, scalar.initial_state(batch.flow_ids))
    second = quantile.prepare(batch, quantile.initial_state(batch.flow_ids))
    assert first.quantiles is None
    assert second.quantiles.shape == (2, 5) and (second.quantiles.diff(dim=1) >= 0).all()
    assert torch.equal(second.point_predictions, median(second.quantiles))
    torch.testing.assert_close(second.working_state, first.working_state, rtol=0, atol=0)
    torch.testing.assert_close(second.detached_tokens, first.detached_tokens, rtol=0, atol=0)
    if variant != "transformer_direct":
        for a, b in zip(
            first.next_state.mac.memory.weights, second.next_state.mac.memory.weights, strict=True
        ):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_median_reproduces_the_scalar_prediction_with_the_same_median_weights():
    scalar, batch = build("mac_online")
    quantile, _ = build("mac_online", head=QUANTILE_HEAD)
    with torch.no_grad():
        quantile.head.weight[MEDIAN_INDEX] = scalar.head.weight[0]
        quantile.head.bias[MEDIAN_INDEX] = scalar.head.bias[0]
    quantile._seal_parameters()
    first = scalar.prepare(batch, scalar.initial_state(batch.flow_ids))
    second = quantile.prepare(batch, quantile.initial_state(batch.flow_ids))
    torch.testing.assert_close(
        second.point_predictions, first.point_predictions, rtol=1e-14, atol=1e-15
    )


def test_pinball_gradients_reach_the_head_and_the_shared_trunk():
    model, batch = build("mac_online", head=QUANTILE_HEAD)
    prepared = model.prepare(batch, model.initial_state(batch.flow_ids), differentiable=True)
    target = torch.tensor([0.01, -0.02], dtype=torch.float64)
    loss = pinball_loss(prepared.quantiles, target)
    names = ("head.weight", "head.bias", "fusion.0.weight", "mac.persistent")
    named = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss, [named[name] for name in names])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    assert gradients[0][[0, 1, 3, 4]].abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("head", ["quantile", "", None, 1])
def test_unknown_heads_are_rejected_by_the_configuration(head):
    with pytest.raises(ValueError, match="cabeza"):
        FinancialConfig(specification(), head=head)


def test_pairing_and_loading_never_cross_heads():
    scalar, _ = build("mac_online", seed=3)
    quantile, _ = build("mac_frozen", head=QUANTILE_HEAD, seed=4)
    with pytest.raises(ValueError):
        copy_paired_parameters(scalar, quantile)
    with pytest.raises(ValueError):
        quantile.load_state_dict(scalar.state_dict())
    other, _ = build("mac_disabled", head=QUANTILE_HEAD, seed=9)
    receipt = copy_paired_parameters(quantile, other)
    assert "head.weight" in receipt["copied_parameters"]
    assert torch.equal(other.head.weight, quantile.head.weight)


def test_episodic_readout_returns_ordered_quantiles_and_their_median():
    model, batch = build("mac_online", head=QUANTILE_HEAD)
    prepared = model.prepare(batch, model.initial_state(batch.flow_ids), differentiable=True)
    extension = reader(refinements=2)
    result = apply_episodic_readout(
        prepared,
        model.head,
        extension,
        snapshot(),
        context_id=CONTEXT,
        cutoff=10,
        differentiable=True,
    )
    assert result.quantiles.shape == (2, 5) and (result.quantiles.diff(dim=1) >= 0).all()
    assert torch.equal(result.point_predictions, result.quantiles[:, MEDIAN_INDEX])
    torch.testing.assert_close(result.quantiles, model.head(result.readout.state), rtol=0, atol=0)
    loss = pinball_loss(result.quantiles, torch.zeros(2, dtype=torch.float64))
    gradients = torch.autograd.grad(
        loss, (model.head.weight, extension.query_projection.weight, model.fusion[0].weight)
    )
    assert all(g.abs().sum() > 0 and torch.isfinite(g).all() for g in gradients)


def test_episodic_none_path_forwards_the_prepared_quantiles():
    model, batch = build("mac_online", head=QUANTILE_HEAD)
    prepared = model.prepare(batch, model.initial_state(batch.flow_ids))
    result = apply_episodic_readout(prepared, model.head)
    assert result.readout is None
    assert result.point_predictions is prepared.point_predictions
    assert result.quantiles is prepared.quantiles


def test_episodic_readout_rejects_heads_that_do_not_match_the_prepared_output():
    scalar, batch = build("mac_online")
    quantile, _ = build("mac_online", head=QUANTILE_HEAD)
    prepared = scalar.prepare(batch, scalar.initial_state(batch.flow_ids))
    with pytest.raises(ValueError, match="cabeza"):
        apply_episodic_readout(prepared, quantile.head)
    with pytest.raises(ValueError, match="cabeza"):
        apply_episodic_readout(
            prepared, quantile.head, reader(), snapshot(), context_id=CONTEXT, cutoff=10
        )
    unordered = torch.nn.Linear(32, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="escalar"):
        apply_episodic_readout(
            prepared, unordered, reader(), snapshot(), context_id=CONTEXT, cutoff=10
        )
    other = quantile.prepare(batch, quantile.initial_state(batch.flow_ids))
    with pytest.raises(ValueError, match="cabeza"):
        apply_episodic_readout(other, scalar.head)


def test_episodic_readout_rejects_nonfinite_quantiles():
    model, batch = build("mac_online", head=QUANTILE_HEAD)
    prepared = model.prepare(batch, model.initial_state(batch.flow_ids))

    class Broken(QuantileHead):
        def forward(self, value):
            result = super().forward(value).clone()
            result[0, 4] = math.inf
            return result

    broken = Broken(32, dtype=torch.float64)
    with pytest.raises(ValueError, match="cuantiles"):
        apply_episodic_readout(
            prepared, broken, reader(), snapshot(), context_id=CONTEXT, cutoff=10
        )


@pytest.mark.parametrize("with_reader", [False, True])
def test_frozen_consumer_accepts_the_common_head_and_keeps_its_quantiles(with_reader):
    model, batch = build("mac_online", head=QUANTILE_HEAD)
    model.eval().requires_grad_(False)
    extension = reader().eval().requires_grad_(False) if with_reader else None
    consumer = FrozenFinancialConsumer(model, readout=extension)
    modes = consumer.identity()["modules"]
    assert modes["predictor/head"]["type"] == "mars_titan.models.quantile_head.QuantileHead"
    assert "mars_titan.models.quantile_head" in consumer.identity()["implementation"]
    bank = EpisodeSnapshot.create(
        **source(),
        cutoff=batch.prediction_at[0],
        codec_id=CODEC,
        context_id=CONTEXT,
        device="cpu",
        dtype=torch.float64,
    )
    options = dict(snapshot=bank) if with_reader else {}
    with torch.no_grad():
        result = consumer.prepare(
            batch, model.initial_state(batch.flow_ids), context_id=CONTEXT, **options
        )
    assert result.quantiles.shape == (2, 5)
    assert torch.equal(result.point_predictions, result.quantiles[:, MEDIAN_INDEX])
    assert json.loads(json.dumps(consumer.identity()))["predictor"]["configuration"]["output"] == (
        QUANTILE_HEAD
    )
