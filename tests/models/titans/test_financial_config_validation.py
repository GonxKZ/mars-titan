"""Precedencia, normalización e identidad de la configuración, sin crear ni ajustar modelos."""

import json
from dataclasses import asdict

import pytest
from test_financial_adapter import specification

from mars_titan.models.titans import financial


def configuration(**changes):
    values = dict(inputs=specification(), variant="mac_online", hidden_size=32, layers=1, seed=42)
    return financial.FinancialConfig(**(values | changes))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            dict(temporal_encoder="unknown", inputs=None),
            "temporal_encoder no identifica un codificador de ventana admitido",
        ),
        (
            dict(inputs=None, head="unknown"),
            "El predictor necesita una entrada y un control identificados",
        ),
        (
            dict(head="unknown", gate_bias={}),
            "La cabeza de salida no pertenece al contrato del predictor",
        ),
        (
            dict(quantile_routing={}, attention_layout="unknown"),
            "quantile_routing declara median_from_tails y encoder_from_spreads",
        ),
        (
            dict(gate_bias={}, memory_stability={}),
            "gate_bias debe declarar alpha_half_life, eta y theta",
        ),
        (
            dict(gate_bias=1, memory_stability={}),
            "memory_stability debe declarar alpha_floor, eta_ceiling, gradient_clip y gate_box",
        ),
        (
            dict(
                memory_plasticity=financial.MemoryPlasticity(capture="eligibility"), sector_map={}
            ),
            "La captura neuronal necesita la cascada multiescala",
        ),
        (
            dict(
                gate_bias=financial.GateBias(),
                memory_stability=financial.MemoryStability(),
                memory_multiscale={},
            ),
            "eta inicial debe ser menor que eta_ceiling",
        ),
        (
            dict(memory_key_competition=1, hidden_size=0),
            "memory_key_competition necesita KeyCompetition, sus valores o None",
        ),
        (
            dict(sector_map={}, session_context="unknown"),
            "sector_map declara solo la carpeta de las fichas en source",
        ),
        (
            dict(presence_modulation="unknown", regional_fusion={}),
            "presence_modulation admite film_v1 o wide_fusion_v1 con la política con máscaras",
        ),
        (
            dict(regional_fusion={}, regional_execution={}),
            "regional_fusion debe declarar toda su configuración",
        ),
        (
            dict(session_rank_inputs="unknown", session_ic_objective=1),
            "session_rank_inputs admite session_rank_inputs_v1 con la política con máscaras",
        ),
        (
            dict(auxiliary_horizons={}, mature_reliability={}),
            "auxiliary_horizons debe declarar horizons, weight y maturity_slack",
        ),
        (
            dict(mature_reliability={}, memory_residual_layer_norm="false"),
            "mature_reliability debe declarar todos sus campos",
        ),
        (
            dict(differential_attention="unknown", memory_residual_layer_norm="false"),
            "differential_attention admite differential_attention_v1 con MAC",
        ),
        (
            dict(memory_residual_layer_norm=1, hidden_size=0),
            "memory_residual_layer_norm debe ser booleano",
        ),
        (
            dict(memory_projections=1, seed=-1),
            "memory_projections debe ser linear_v1 o titans_mac_paper_projections_v2",
        ),
    ],
)
def test_multiple_invalid_fields_keep_the_first_exception_and_its_message(changes, message):
    assert configuration().variant == "mac_online"
    with pytest.raises(ValueError) as raised:
        configuration(**changes)
    assert type(raised.value) is ValueError
    assert str(raised.value) == message


@pytest.mark.parametrize(
    "changes",
    [
        dict(gate_bias=financial.GateBias()),
        dict(gate_bias=financial.GateBias(eta=0.1), memory_stability=financial.MemoryStability()),
        dict(memory_multiscale=financial.MemoryMultiscale()),
        dict(
            memory_multiscale=financial.MemoryMultiscale(),
            memory_plasticity=financial.MemoryPlasticity(capture="eligibility"),
        ),
        dict(memory_key_competition=financial.KeyCompetition(center=True, retained=8)),
        dict(head=financial.QUANTILE_HEAD, quantile_routing=financial.QuantileRouting()),
        dict(auxiliary_horizons=financial.AuxiliaryHorizons()),
        dict(mature_reliability=financial.MatureReliability()),
        dict(
            memory_multiscale=financial.MemoryMultiscale(form="ema_bank"),
            memory_bank_readout=financial.BankReadoutConfig(),
        ),
        dict(regional_fusion=financial.RegionalFusionConfig()),
        dict(cross_asset_attention=financial.CrossAssetAttentionConfig()),
        dict(memory_huber=financial.HuberMemory()),
        dict(gate_bias=financial.GateBias(), memory_time_scales=financial.ScaleInvariantTime()),
    ],
)
def test_typed_and_json_components_keep_the_same_normalized_configuration_and_identity(changes):
    raw = {
        name: asdict(value) if hasattr(value, "__dataclass_fields__") else value
        for name, value in changes.items()
    }
    typed, parsed = configuration(**changes), configuration(**json.loads(json.dumps(raw)))
    assert typed == parsed
    assert typed.identity() == parsed.identity()
    assert asdict(typed) == asdict(parsed)
