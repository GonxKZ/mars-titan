"""Posición de la atención entre activos respecto a la rama técnica.

Por defecto C5 mezcla la salida de MAC, antes de que la rama técnica sume su residuo, y no ve
esa señal. Con `position="after_technical"` mezcla la representación completa. Las pruebas
calculan salidas sobre lotes ficticios, sin optimizador ni pasos.
"""

from dataclasses import asdict

import pytest
import torch
from test_financial_learning_scale import build, emit
from test_technical_dense import DENSE, opened

from mars_titan.models.titans.cross_asset_attention import (
    CrossAssetAttentionConfig,
    parse_cross_asset_attention,
)
from mars_titan.training.search_cases import checked_predictor_changes

EARLY = CrossAssetAttentionConfig()
LATE = CrossAssetAttentionConfig(position="after_technical")
LATE_CONTROL = CrossAssetAttentionConfig(mode="self_only", position="after_technical")


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def predictor(attention, *, weight=0.05):
    """Predictor con la rama densa abierta y, si hay C5, su proyección de salida no nula."""
    options = {} if attention is None else dict(cross_asset_attention=asdict(attention))
    model, batch = build(technical_branch=asdict(DENSE), **options)
    opened(model.technical_branch)
    if attention is not None and weight:
        with torch.no_grad():
            model.cross_asset_attention.out.weight.fill_(weight)
    model._seal_parameters()
    return model, batch


def session(batch):
    return tuple(flow.split("/", 1)[0] for flow in batch.flow_ids), batch.prediction_at


def test_the_default_position_keeps_its_declaration_and_identity():
    assert asdict(EARLY) == dict(heads=4, mode="same_session", position="after_mac")
    assert EARLY.identity()["position"] == "after_mac_before_head"
    assert LATE.identity()["position"] == "after_technical_branch_before_head"
    # La declaración de dos campos de las campañas anteriores sigue valiendo.
    assert parse_cross_asset_attention(dict(heads=4, mode="same_session")) == EARLY
    assert parse_cross_asset_attention(asdict(LATE)) == LATE
    changes = dict(cross_asset_attention=asdict(LATE))
    assert checked_predictor_changes(changes) == changes
    for wrong in ("before_mac", None, 1):
        with pytest.raises(ValueError, match="posición"):
            CrossAssetAttentionConfig(position=wrong)
    with pytest.raises(ValueError):
        parse_cross_asset_attention(dict(heads=4, mode="same_session", place="after_technical"))


@pytest.mark.parametrize("config", [LATE, LATE_CONTROL])
def test_the_late_position_starts_as_the_reference(config):
    plain, batch = predictor(None)
    late, _ = predictor(config, weight=0)
    expected = emit(plain, batch)
    actual = emit(late, batch, session_complete=True)
    assert torch.count_nonzero(late.cross_asset_attention.out.weight) == 0
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.working_state, expected.working_state, rtol=0, atol=0)


def test_the_late_position_mixes_the_representation_with_its_technical_residual():
    plain, batch = predictor(None)
    late, _ = predictor(LATE)
    early, _ = predictor(EARLY)
    markets, instants = session(batch)
    reference = emit(plain, batch)
    # Sin la rama, la representación es la salida de MAC: la diferencia es el residuo técnico.
    bare, _ = build()
    from_mac = emit(bare, batch).working_state
    residual = reference.working_state - from_mac
    assert torch.count_nonzero(residual) > 0
    after = emit(late, batch, session_complete=True)
    # Tardía: C5 lee lo que sale de MAC más el residuo técnico, que es lo que ve la cabeza.
    expected = late.cross_asset_attention(reference.working_state, markets, instants)
    torch.testing.assert_close(after.working_state, expected, rtol=0, atol=0)
    # Temprana: C5 lee solo lo que sale de MAC y el residuo se suma después.
    before = emit(early, batch, session_complete=True)
    mixed = early.cross_asset_attention(from_mac, markets, instants)
    torch.testing.assert_close(before.working_state, mixed + residual, rtol=1e-12, atol=1e-12)
    # La mediana nace en cero, así que las dos posiciones se distinguen en la representación.
    assert not torch.allclose(after.working_state, before.working_state)
    # Ninguna de las dos toca la memoria.
    for emitted in (after, before):
        torch.testing.assert_close(
            emitted.next_state.mac.memory.weights[0],
            reference.next_state.mac.memory.weights[0],
            rtol=0,
            atol=0,
        )


def gradients(model, batch, names):
    state = model.initial_state(batch.flow_ids, differentiable=True)
    emitted = model.prepare(batch, state, differentiable=True, session_complete=True)
    parameters = dict(model.named_parameters())
    # La suma de la representación que lee la cabeza: su gradiente respecto al residuo técnico
    # es uno si el residuo se suma al final, y pasa por C5 si C5 lo mezcla. La cabeza de
    # cuantiles no es lineal y mezclaría el camino con el punto de trabajo.
    values = torch.autograd.grad(
        emitted.working_state.sum(), [parameters[name] for name in names], allow_unused=True
    )
    return dict(zip(names, values, strict=True))


def test_the_gradient_crosses_the_late_mixer_towards_the_branch():
    names = (
        "cross_asset_attention.out.weight",
        "cross_asset_attention.qkv.weight",
        "technical_branch.outer.weight",
    )
    # Al construirse, W_o es cero: solo ella recibe gradiente de C5 y la rama, el de siempre.
    fresh, batch = predictor(LATE, weight=0)
    start = gradients(fresh, batch, names)
    assert start["cross_asset_attention.out.weight"].abs().sum() > 0
    assert start["cross_asset_attention.qkv.weight"].abs().sum() == 0
    plain, _ = predictor(None)
    direct = gradients(plain, batch, names[2:])["technical_branch.outer.weight"]
    torch.testing.assert_close(start["technical_branch.outer.weight"], direct, rtol=0, atol=0)
    # Con W_o abierta, consultas y claves aprenden y la rama recibe además lo que cruza C5.
    late, _ = predictor(LATE)
    opened_ = gradients(late, batch, names)
    assert opened_["cross_asset_attention.qkv.weight"].abs().sum() > 0
    through = opened_["technical_branch.outer.weight"]
    assert not torch.allclose(through, direct)
    # En la posición de partida la rama no pasa por C5: su gradiente es el de la cabeza.
    early, _ = predictor(EARLY)
    before = gradients(early, batch, names)["technical_branch.outer.weight"]
    torch.testing.assert_close(before, direct, rtol=1e-12, atol=1e-12)


def test_the_late_position_needs_the_technical_branch():
    with pytest.raises(ValueError, match="rama técnica"):
        build(cross_asset_attention=asdict(LATE))
    build(cross_asset_attention=asdict(EARLY))


def test_the_two_positions_are_different_identities_with_the_same_parameters():
    early, _ = predictor(EARLY)
    late, _ = predictor(LATE)
    assert early.config.identity() != late.config.identity()
    assert early._config_id() != late._config_id()
    for (name, a), (_, b) in zip(early.named_parameters(), late.named_parameters(), strict=True):
        assert torch.equal(a, b), name
    with pytest.raises(ValueError):
        late.load_state_dict(early.state_dict())
