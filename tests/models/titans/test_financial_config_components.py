"""Rechazo de declaraciones malformadas y fronteras del presupuesto de configuración."""

import pytest
from test_financial_config_validation import configuration

from mars_titan.models.titans import financial


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("head", "unknown"),
        ("gate_bias", 1),
        ("memory_multiscale", 1),
        ("memory_plasticity", 1),
        ("memory_key_competition", 1),
        ("session_context", "unknown"),
        ("presence_modulation", "unknown"),
        ("regional_fusion", 1),
        ("regional_execution", 1),
        ("regional_variant", 1),
        ("holographic_network", 1),
        ("memory_bank_readout", 1),
        ("session_rank_inputs", "unknown"),
        ("cross_asset_attention", 1),
        ("auxiliary_horizons", 1),
        ("mature_reliability", 1),
        ("memory_projections", "unknown"),
        ("max_batch", 0),
        ("axial_time_region", 1),
        ("fractal_shared_operator", 1),
        ("memory_huber", 1),
        ("session_ic_objective", 1),
    ],
)
def test_each_component_rejects_its_malformed_declaration_before_use(field, value):
    assert configuration().variant == "mac_online"
    with pytest.raises(ValueError):
        configuration(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 0),
        ("seed", 2**32 - 1),
        ("persistent_tokens", 0),
        ("persistent_tokens", 64),
        ("max_batch", 1),
        ("max_batch", financial.MAX_BLOCK_ROWS),
        ("max_state_bytes", 1),
        ("max_state_bytes", financial.MAX_STATE_BYTES),
    ],
)
def test_budget_boundaries_remain_admissible_without_allocating_their_state(field, value):
    config = configuration(**{field: value})
    assert getattr(config, field) == value
    assert type(getattr(config, field)) is int


@pytest.mark.parametrize(
    ("field", "component"),
    [
        ("memory_plasticity", financial.MemoryPlasticity()),
        ("memory_stability", financial.MemoryStability()),
    ],
)
def test_a_reader_and_a_write_intervention_remain_separate_comparisons(field, component):
    reader = dict(
        memory_multiscale=financial.MemoryMultiscale(form="ema_bank"),
        memory_bank_readout=financial.BankReadoutConfig(),
    )
    assert configuration(**reader).memory_bank_readout is not None
    assert getattr(configuration(**{field: component}), field) == component
    with pytest.raises(ValueError, match="El lector con plasticidad o PT1 necesita otro contraste"):
        configuration(**reader, **{field: component})
