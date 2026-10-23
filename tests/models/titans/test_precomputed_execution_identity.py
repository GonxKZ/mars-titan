"""Rechazar un backend nominal distinto del cálculo instalado antes de sellarlo."""

from dataclasses import replace

import pytest


@pytest.mark.parametrize("tamper", ["top", "block"])
def test_standalone_encoder_rejects_inconsistent_execution_before_encoding(tamper):
    from tests.test_kronlstm import encoder, prices

    model = encoder(execution="precomputed").eval()
    if tamper == "top":
        model.execution = "reference"
    else:
        model.blocks[0].execution = "reference"
    with pytest.raises(ValueError):
        model.encode_last(prices("cpu", "float64"))


@pytest.mark.parametrize(
    "declared,installed",
    [("kronlstm_precomputed_v1", "reference"), ("kronlstm_v1", "precomputed")],
)
def test_frozen_consumer_rejects_a_coherent_execution_substitution_before_sealing(
    declared, installed
):
    import torch
    from test_financial_adapter import api, setup

    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    original, _ = setup("transformer_direct")
    model = api().FinancialPredictor(
        replace(original.config, temporal_encoder=declared), dtype=torch.float64
    )
    model.eval().requires_grad_(False)
    model.price_encoder.execution = installed
    for block in model.price_encoder.blocks:
        block.execution = installed
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with pytest.raises(ValueError):
            FrozenFinancialConsumer(model)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
