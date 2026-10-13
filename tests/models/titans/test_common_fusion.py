"""Codificación y fusión comunes a los cuatro controles financieros, sin optimizadores.

Las comprobaciones usan lotes técnicos y retropropagan una pérdida cuadrática de la
salida para ver por dónde llega el gradiente. No se aplica ningún paso de ajuste.
"""

import pytest
import torch

from mars_titan.models.baselines.multimodal import (
    MODALITIES,
    PRESENCE_FUSION,
    STRICT_FUSION,
    MultimodalReference,
)
from tests.models.titans.test_financial_adapter import (
    DIMENSIONS,
    VARIANTS,
    api,
    raw_batch,
    setup,
    specification,
)

SHARED = ("price_encoder", "encoders", "fusion", "head")


def shared_state(model):
    return {
        name: value for name, value in model.state_dict().items() if name.split(".", 1)[0] in SHARED
    }


@pytest.mark.parametrize("historical", [True, False])
def test_the_four_controls_share_the_common_modules_and_emit_the_same_token(historical):
    built = {variant: setup(variant, historical=historical) for variant in VARIANTS}
    direct, batch = built["transformer_direct"]
    expected = shared_state(direct)
    prepared = direct.prepare(batch, direct.initial_state(batch.flow_ids))
    token = prepared.detached_tokens
    # Sin memoria, la cabeza lee el token fusionado sin ninguna otra transformación.
    with torch.no_grad():
        head = direct.head(token).squeeze(-1)
    torch.testing.assert_close(prepared.point_predictions, head, rtol=0, atol=0)
    for variant, (model, other) in built.items():
        # Mismas muestras, mismas modalidades y mismos bits de presencia en cada control.
        assert other.sample_ids == batch.sample_ids
        assert other.input_digest == batch.input_digest
        actual = shared_state(model)
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            # Algunos módulos guardan su identidad como estado extra en un diccionario.
            same = (
                torch.equal(actual[name], value)
                if torch.is_tensor(value)
                else actual[name] == value
            )
            assert same, (variant, name)
        # Lo único que añade un control con memoria es el módulo MAC.
        extra = {name.split(".", 1)[0] for name in model.state_dict()} - set(SHARED)
        assert extra == (
            {"_extra_state"} if variant == "transformer_direct" else {"_extra_state", "mac"}
        )
        result = model.prepare(other, model.initial_state(other.flow_ids))
        torch.testing.assert_close(result.detached_tokens, token, rtol=0, atol=0)


def test_each_present_modality_reaches_its_projection_and_an_absent_one_gets_no_gradient():
    spec = specification()
    config = api().FinancialConfig(spec, variant="mac_online", hidden_size=32, layers=1, seed=42)
    model = api().FinancialPredictor(config, dtype=torch.float64)
    raw = raw_batch()
    # Noticias ausentes en las dos filas. El lector deja su relleno a cero.
    raw["presence"][:, MODALITIES.index("news")] = False
    raw["inputs"]["news"][:] = 0
    batch = api().DecisionBatch.from_corpus(raw, spec, dtype=torch.float64)
    state = model.initial_state(batch.flow_ids, differentiable=True)
    result = model.prepare(batch, state, differentiable=True)
    result.point_predictions.square().sum().backward()
    for parameter in model.encoders["news"].parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0
    for name in ("charts", "fundamentals", "macro"):
        for parameter in model.encoders[name].parameters():
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    assert model.price_encoder.projection.weight.grad.abs().sum() > 0
    # Las columnas de los bits de presencia también reciben gradiente.
    assert model.fusion[0].weight.grad[:, -len(MODALITIES) :].abs().sum() > 0


@pytest.mark.parametrize("historical", [True, False])
def test_reference_and_predictor_name_the_same_fusion(historical):
    model, _ = setup("mac_online", historical=historical)
    fusion = PRESENCE_FUSION if historical else STRICT_FUSION
    assert model.config.identity()["mask_fusion"] == fusion
    reference = MultimodalReference(
        "transformer", DIMENSIONS, context=64, hidden_size=32, mask_fusion=fusion
    )
    # La referencia solo declara el campo cuando la fusión no es la estricta anterior.
    assert reference.configuration.get("mask_fusion", STRICT_FUSION) == fusion
    assert model.fusion[0].in_features == reference.fusion[0].in_features
