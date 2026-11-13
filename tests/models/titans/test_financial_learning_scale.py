"""Escalas de aprendizaje y codificadores de referencia en el predictor financiero (#836).

Lotes técnicos en CPU. Solo se calculan pasadas hacia delante y gradientes, nunca pasos de
optimizador.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_financial_adapter import DIMENSIONS, VARIANTS, raw_batch, specification

from mars_titan.models.baselines.multimodal import (
    PRESENCE_FUSION,
    STRICT_FUSION,
    MultimodalReference,
)
from mars_titan.models.learning_scale import (
    INITIAL_INTERVAL,
    TRANSFORM,
    LearningScale,
    fit_statistics,
)
from mars_titan.models.quantile_head import (
    MEDIAN_INDEX,
    QUANTILE_HEAD,
    QuantileHead,
    ScaledLinear,
    ordered_quantiles,
    pinball_loss,
)
from mars_titan.models.titans.episodic_readout import head_predictions
from mars_titan.models.titans.financial import (
    REFERENCE_CONTRACT,
    REFERENCE_ENCODERS,
    DecisionBatch,
    FinancialConfig,
    FinancialPredictor,
    ReferencePriceEncoder,
    copy_paired_parameters,
)
from mars_titan.models.titans.session_ic_objective import SessionICObjective
from mars_titan.models.titans.session_ranks import SESSION_RANK_INPUTS
from mars_titan.training.search_cases import checked_predictor_changes

DECLARED = dict(clip=5.0, price_returns=True, zero_median=True)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def statistics(config, *, rows=200, seed=3, label=0.02):
    """Estadísticas de un tramo técnico con las dimensiones del lote de las pruebas."""
    generator = np.random.default_rng(seed)
    inputs = {
        name: generator.normal(
            1.5, 2.0, size=(rows, 64, width) if name == "prices" else (rows, width)
        ).astype(np.float32)
        for name, width in DIMENSIONS.items()
    }
    for name in ("fundamentals", "macro"):
        inputs[name][:, 1] = 1
        inputs[name][:, 2] = np.abs(inputs[name][:, 2])
    presence = np.ones((rows, 5), dtype=np.bool_)
    return fit_statistics(config, inputs, presence, generator.normal(scale=label, size=rows))


def build(variant="mac_online", *, head=QUANTILE_HEAD, historical=True, fitted=True, **options):
    spec = specification(historical=historical)
    options.setdefault("learning_scale", DECLARED)
    config = FinancialConfig(
        spec, variant=variant, hidden_size=32, layers=1, seed=42, head=head, **options
    )
    model = FinancialPredictor(config, dtype=torch.float64)
    if fitted and config.learning_scale is not None:
        model.set_learning_statistics(statistics(config.learning_scale))
    raw = raw_batch(absent=historical)
    if not historical:
        raw.pop("presence")
    return model, DecisionBatch.from_corpus(raw, spec, dtype=torch.float64)


def emit(model, batch, **options):
    return model.prepare(batch, model.initial_state(batch.flow_ids), **options)


def test_without_the_declaration_the_predictor_keeps_its_identity_and_state():
    plain, _ = build(learning_scale=None)
    identity = plain.config.identity()
    assert "learning_scale" not in identity and "reference_encoder" not in identity
    assert plain.learning_scaler is None and type(plain.head) is QuantileHead
    assert not plain.head.scaled
    assert not [name for name in plain.state_dict() if "learning_scaler" in name or "scale" in name]
    scalar, _ = build(head="scalar", learning_scale=None)
    assert type(scalar.head) is torch.nn.Linear
    declared, _ = build()
    assert declared.config.identity() == {
        **identity,
        "learning_scale": LearningScale(**DECLARED).identity(),
    }
    assert declared.config.identity()["learning_scale"]["name"] == TRANSFORM
    assert declared._config_id() != plain._config_id()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("head", ["scalar", QUANTILE_HEAD])
@pytest.mark.parametrize("historical", [True, False])
def test_scaled_predictor_starts_at_zero_with_label_sized_quantiles_and_the_same_trunk(
    variant, head, historical
):
    plain, batch = build(variant, head=head, historical=historical, learning_scale=None)
    model, _ = build(variant, head=head, historical=historical)
    left, right = dict(plain.named_parameters()), dict(model.named_parameters())
    assert left.keys() == right.keys()
    row = MEDIAN_INDEX if head == QUANTILE_HEAD else 0
    for name, value in left.items():
        if name == "head.weight":
            keep = [index for index in range(len(value)) if index != row]
            assert torch.equal(value[keep], right[name][keep])
            assert torch.equal(right[name][row], torch.zeros_like(value[row]))
        elif name == "head.bias":
            assert right[name][row] == 0
        else:
            assert torch.equal(value, right[name]), name
    assert type(model.head) is (QuantileHead if head == QUANTILE_HEAD else ScaledLinear)
    result = emit(model, batch)
    assert torch.equal(result.point_predictions, torch.zeros(2, dtype=torch.float64))
    scale = float(model.head.output_scale)
    assert scale == pytest.approx(statistics(model.config.learning_scale)["label_scale"])
    if head == QUANTILE_HEAD:
        assert (result.quantiles.diff(dim=1) >= 0).all()
        spread = result.quantiles[:, 3] - result.quantiles[:, 1]
        # Con incrementos softplus cercanos a ln 2, el intervalo 10-90 es el de las etiquetas.
        assert (spread > 0.5 * INITIAL_INTERVAL * scale).all()
        assert (spread < 2.0 * INITIAL_INTERVAL * scale).all()
    else:
        assert result.quantiles is None


def test_sampled_median_keeps_the_drawn_head_and_only_scales_its_output():
    declared = dict(DECLARED, zero_median=False)
    plain, batch = build(learning_scale=None)
    model, _ = build(learning_scale=declared)
    for name, value in plain.named_parameters():
        assert torch.equal(value, dict(model.named_parameters())[name]), name
    assert emit(model, batch).point_predictions.abs().sum() > 0
    scalar, _ = build(head="scalar", learning_scale=declared)
    reference, _ = build(head="scalar", learning_scale=None)
    assert torch.equal(scalar.head.weight, reference.head.weight)
    assert torch.equal(scalar.head.bias, reference.head.bias)


def test_prepare_refuses_until_the_fit_statistics_arrive():
    model, batch = build(fitted=False)
    with pytest.raises(ValueError, match="Faltan las estadísticas"):
        emit(model, batch)
    with pytest.raises(ValueError, match="Faltan las estadísticas"):
        model.fused_tokens(batch.inputs, batch.presence)
    values = statistics(model.config.learning_scale)
    for wrong in (None, {**values, "label_scale": 0.0}, {**values, "label_scale": 1}):
        with pytest.raises(ValueError, match="escala positiva"):
            model.set_learning_statistics(wrong)
    with pytest.raises(ValueError, match="no corresponden"):
        model.set_learning_statistics({**values, "transform": {}})
    model.set_learning_statistics(values)
    assert emit(model, batch).quantiles.shape == (2, 5)
    plain, _ = build(learning_scale=None)
    with pytest.raises(ValueError, match="no declara learning_scale"):
        plain.set_learning_statistics(values)


@pytest.mark.parametrize("head", ["scalar", QUANTILE_HEAD])
def test_output_is_the_head_output_times_the_label_scale(head):
    declared = dict(DECLARED, zero_median=False)
    small, batch = build(head=head, learning_scale=declared, fitted=False)
    large, _ = build(head=head, learning_scale=declared, fitted=False)
    config = small.config.learning_scale
    small.set_learning_statistics(statistics(config, label=0.02))
    large.set_learning_statistics(statistics(config, label=0.06))
    ratio = float(large.head.output_scale / small.head.output_scale)
    assert ratio == pytest.approx(3.0, rel=0.35) and ratio != 1.0
    first, second = emit(small, batch), emit(large, batch)
    # Las entradas se escalan igual, así que el estado de trabajo coincide y solo cambia la salida.
    assert torch.equal(first.working_state, second.working_state)
    torch.testing.assert_close(second.point_predictions, first.point_predictions * ratio)
    if head == QUANTILE_HEAD:
        torch.testing.assert_close(second.quantiles, first.quantiles * ratio)
        raw = torch.nn.functional.linear(first.working_state, small.head.weight, small.head.bias)
        torch.testing.assert_close(
            first.quantiles, ordered_quantiles(raw) * small.head.output_scale
        )
        # El lector episódico aplica la misma cabeza, con su escala.
        predictions, quantiles = head_predictions(small.head, first.working_state)
        assert torch.equal(quantiles, first.quantiles)
        assert torch.equal(predictions, first.point_predictions)


def test_inputs_reach_the_trunk_scaled_and_the_received_batch_is_not_modified():
    plain, batch = build(learning_scale=None)
    model, _ = build()
    before = {name: value.clone() for name, value in batch.inputs.items()}
    scaled = emit(model, batch)
    for name, value in batch.inputs.items():
        assert torch.equal(value, before[name])
    assert not torch.equal(scaled.detached_tokens, emit(plain, batch).detached_tokens)
    # El token es el de las entradas normalizadas por el escalador.
    expected = plain.fused_tokens(
        model.learning_scaler(batch.inputs, batch.presence), batch.presence
    )
    torch.testing.assert_close(scaled.detached_tokens, expected, rtol=0, atol=0)
    assert torch.equal(model.fused_tokens(batch.inputs, batch.presence), scaled.detached_tokens)


def test_statistics_enter_the_parameter_identity_and_travel_in_the_state():
    model, batch = build(fitted=False)
    unfitted = model._parameter_id
    model.set_learning_statistics(statistics(model.config.learning_scale))
    assert model._parameter_id != unfitted
    model.verify_parameter_identity()
    other, _ = build(fitted=False)
    other.set_learning_statistics(statistics(model.config.learning_scale, seed=4))
    assert other._parameter_id != model._parameter_id
    twin, _ = build(fitted=False)
    twin.load_state_dict(model.state_dict())
    assert twin._parameter_id == model._parameter_id and bool(twin.learning_scaler.fitted)
    assert torch.equal(emit(twin, batch).quantiles, emit(model, batch).quantiles)
    # Un estado sin las estadísticas no completa el contrato del predictor escalado.
    plain, _ = build(learning_scale=None)
    with pytest.raises(ValueError):
        twin.load_state_dict(plain.state_dict())


def test_paired_controls_share_statistics_and_initial_parameters():
    source, batch = build("mac_online")
    target, _ = build("transformer_direct")
    receipt = copy_paired_parameters(source, target)
    assert receipt["target_after"] == target._parameter_id
    assert torch.equal(source.head.output_scale, target.head.output_scale)
    assert torch.equal(source.learning_scaler.macro_center, target.learning_scaler.macro_center)
    assert torch.equal(emit(target, batch).point_predictions, torch.zeros(2, dtype=torch.float64))
    unscaled, _ = build("transformer_direct", learning_scale=None)
    with pytest.raises(ValueError):
        copy_paired_parameters(source, unscaled)


def test_gradients_reach_head_trunk_and_memory_through_the_scaled_path():
    model, batch = build("mac_online")
    prepared = emit(model, batch, differentiable=True)
    loss = pinball_loss(prepared.quantiles, torch.tensor([0.01, -0.02], dtype=torch.float64))
    names = ("head.weight", "head.bias", "fusion.0.weight", "mac.persistent")
    named = dict(model.named_parameters())
    gradients = torch.autograd.grad(
        loss, [named[name] for name in names], allow_unused=True, retain_graph=True
    )
    # La fila de la mediana vale cero, pero su gradiente no: recibe la señal desde el primer paso.
    assert gradients[0][MEDIAN_INDEX].abs().sum() > 0
    assert all(g is not None and torch.isfinite(g).all() for g in gradients[:2])
    spread = pinball_loss(prepared.quantiles, torch.tensor([0.5, -0.5], dtype=torch.float64))
    trunk = torch.autograd.grad(spread, [named["fusion.0.weight"], named["mac.persistent"]])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in trunk)


@pytest.mark.parametrize(
    "options",
    [
        dict(session_rank_inputs=SESSION_RANK_INPUTS),
        dict(head="scalar", session_ic_objective=SessionICObjective()),
    ],
)
def test_learning_scale_is_not_combined_with_session_ranks_or_output_objectives(options):
    spec = specification()
    head = options.pop("head", QUANTILE_HEAD)
    FinancialConfig(spec, variant="mac_online", hidden_size=32, head=head, **options)
    with pytest.raises(ValueError, match="no se combina"):
        FinancialConfig(
            spec,
            variant="mac_online",
            hidden_size=32,
            head=head,
            learning_scale=DECLARED,
            **options,
        )


def test_learning_scale_is_declared_complete_and_admitted_as_a_predictor_change():
    spec = specification()
    for wrong in (dict(clip=5.0), "learning_scale_v1", dict(DECLARED, clip=0.0)):
        with pytest.raises(ValueError):
            FinancialConfig(spec, variant="mac_online", learning_scale=wrong)
    changes = dict(learning_scale=DECLARED, reference_encoder="gru")
    assert checked_predictor_changes(changes) == changes
    for wrong in (dict(learning_scale="v1"), dict(learning_scale={}), dict(reference_encoder={})):
        with pytest.raises(ValueError, match="cambios del predictor"):
            checked_predictor_changes(wrong)


@pytest.mark.parametrize("kind", REFERENCE_ENCODERS)
@pytest.mark.parametrize("historical", [True, False])
@pytest.mark.parametrize("head", ["scalar", QUANTILE_HEAD])
def test_reference_encoder_reproduces_the_reference_model(kind, historical, head):
    """El predictor con el codificador de una referencia calcula lo mismo que esa referencia."""
    model, batch = build(
        "transformer_direct",
        head=head,
        historical=historical,
        learning_scale=None,
        reference_encoder=kind,
    )
    assert type(model.price_encoder) is ReferencePriceEncoder and model.mac is None
    assert model.config.identity()["reference_encoder"] == dict(
        kind=kind, contract=REFERENCE_CONTRACT
    )
    assert model.get_extra_state()["backbone_contract"] == dict(
        kind=kind, contract=REFERENCE_CONTRACT
    )
    reference = MultimodalReference(
        kind,
        DIMENSIONS,
        context=64,
        hidden_size=32,
        layers=1,
        mask_fusion=PRESENCE_FUSION if historical else STRICT_FUSION,
        head=head,
    ).double()
    reference.price_encoder.load_state_dict(model.price_encoder.encoder.state_dict())
    reference.encoders.load_state_dict(model.encoders.state_dict())
    reference.fusion.load_state_dict(model.fusion.state_dict())
    reference.head.load_state_dict(model.head.state_dict())
    with torch.no_grad():
        expected = reference(batch.inputs, batch.presence if historical else None)
    result = emit(model, batch)
    emitted = result.quantiles if head == QUANTILE_HEAD else result.point_predictions
    torch.testing.assert_close(emitted, expected, rtol=1e-12, atol=1e-14)
    # La copia del mismo predictor con otra semilla sortea otros pesos.
    other = FinancialPredictor(replace(model.config, seed=43), dtype=torch.float64)
    assert not torch.equal(other.fusion[0].weight, model.fusion[0].weight)


@pytest.mark.parametrize("kind", REFERENCE_ENCODERS)
def test_reference_encoder_works_with_learning_scale_and_keeps_its_state(kind):
    model, batch = build("transformer_direct", reference_encoder=kind)
    result = emit(model, batch)
    assert torch.equal(result.point_predictions, torch.zeros(2, dtype=torch.float64))
    twin, _ = build("transformer_direct", reference_encoder=kind, fitted=False)
    twin.load_state_dict(model.state_dict())
    assert torch.equal(emit(twin, batch).quantiles, result.quantiles)
    loss = pinball_loss(
        emit(model, batch, differentiable=True).quantiles,
        torch.tensor([0.5, -0.5], dtype=torch.float64),
    )
    gradients = torch.autograd.grad(loss, list(model.price_encoder.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_reference_encoder_is_one_of_the_references_without_another_temporal_encoder():
    spec = specification()
    for wrong in ("transformer", "tcn", 1, ""):
        with pytest.raises(ValueError, match="rnn, lstm, gru o dlinear"):
            FinancialConfig(spec, variant="transformer_direct", reference_encoder=wrong)
    with pytest.raises(ValueError, match="sin otro codificador temporal"):
        FinancialConfig(
            spec,
            variant="transformer_direct",
            reference_encoder="gru",
            temporal_encoder="kronlstm_v1",
        )
    with pytest.raises(ValueError, match="no pertenece a las referencias"):
        ReferencePriceEncoder("transformer", torch.nn.Identity())
