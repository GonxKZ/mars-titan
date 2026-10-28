"""Contratos de formas y configuración de las transformaciones por sesión."""

from dataclasses import replace

import pytest
import torch
from test_financial_adapter import setup

from mars_titan.models.titans.session_ic_objective import (
    SessionICObjective,
    SessionScale,
    session_correlations,
    session_ic_loss,
    standardized_scores,
)
from mars_titan.models.titans.session_ranks import gaussian_ranks, rank_session_inputs


@pytest.mark.parametrize(
    "values, present",
    [
        ([], torch.ones(2, 1, dtype=torch.bool)),
        (torch.ones(2, 1), None),
        (torch.ones(2), torch.ones(2, dtype=torch.bool)),
        (torch.ones(2, 1), torch.ones(2, dtype=torch.bool)),
        (torch.ones(2, 1), torch.ones(2, 1)),
        (torch.ones(2, 1, dtype=torch.int64), torch.ones(2, 1, dtype=torch.bool)),
    ],
)
def test_rank_contract_rejects_nonmatrix_or_unaligned_presence(values, present):
    with pytest.raises(ValueError, match="rangos"):
        gaussian_ranks(values, present)


def test_empty_rank_input_keeps_its_shape_precision_and_device():
    values = torch.empty(0, 3, dtype=torch.float64)
    result = gaussian_ranks(values, torch.empty(0, 3, dtype=torch.bool))
    assert result.shape == values.shape and result.dtype == values.dtype
    assert result.device == values.device


def test_session_ranks_reject_unknown_price_layout_and_incomplete_fundamentals():
    with pytest.raises(ValueError, match="canales"):
        rank_session_inputs(dict(prices=torch.ones(2, 64, 4)), ["US", "US"])
    for markets, numeric in [(["US"], torch.ones(2, 3)), (["US", "US"], torch.ones(2, 2))]:
        with pytest.raises(ValueError, match="fundamentales"):
            rank_session_inputs(dict(prices=torch.ones(2, 64, 5), fundamentals=numeric), markets)


@pytest.mark.parametrize("field", ["initial_scale", "scale_weight"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_scale_configuration_rejects_nonfinite_numbers(field, value):
    with pytest.raises(ValueError, match="finito"):
        SessionICObjective(**{field: value})


def test_loss_and_standardization_reject_misaligned_rows():
    with pytest.raises(ValueError, match="mercado"):
        standardized_scores(torch.ones(3), ["US"])
    with pytest.raises(ValueError, match="fila a fila"):
        session_correlations(torch.ones(3), torch.ones(2), [0] * 3, 3)
    with pytest.raises(ValueError, match="pérdida de IC"):
        session_ic_loss(torch.ones(3, 3), torch.ones(3), [0] * 3, SessionICObjective())


def test_predictor_and_scale_reject_an_unidentified_objective():
    with pytest.raises(ValueError, match="configuración"):
        SessionScale({"minimum_rows": 8})
    base, _ = setup()
    with pytest.raises(ValueError, match="configuración"):
        replace(base.config, session_ic_objective=True)
